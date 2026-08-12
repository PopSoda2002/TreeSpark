"""Real dynamic-batching mini-engine for TreeSpark (greedy, single GPU).

The batch is PHYSICALLY composed on the GPU every step: active requests' trees are
padded to a common length and verified in ONE target forward; requests join at
admission (per-request prefill) and leave on completion; bs fluctuates.

KV design: per-row region of a preallocated static buffer; ALL tree tokens' KV are
written at fresh offsets and never moved -- rejected nodes simply never enter the
row's valid-set, so the per-step 4D mask (valid prefix + tree ancestors) is the
single source of truth. No compaction, no KV copies.

Step input per row: [anchor] + tree tokens (anchor's KV is written this step; its
hidden joins the drafter features when the round commits).

Correctness gate: at bs=1 this reproduces dspark_tree_generate token-exactly.
"""

import json
import math
import os
import time

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

from treespark.dspark import DSparkDraft, dspark_context_features
from treespark.sampling import probs_from_logits, sample_from_probs
from treespark.tree_speculative import walk_tree

from treespark import tree as TS

TARGET = "Qwen/Qwen3-4B"
DSPARK_DIR = "checkpoints/dspark_qwen3_4b_block7"
DATA_DIR = "eval_datasets"
MAX_BS, MAX_LEN, MAX_NEW = 16, 1600, 100
PLATT_A, PLATT_B = 0.6684, -0.4587
NEG = torch.finfo(torch.bfloat16).min


class RaggedCache(Cache):
    """Static per-row-offset KV buffer; `write_idx` [bs, t] set before each forward."""

    def __init__(self, config, bs, max_len, device, dtype):
        self.n_layers = config.num_hidden_layers
        kvh = config.num_key_value_heads
        hd = config.head_dim
        self.kbuf = [torch.zeros(bs, kvh, max_len, hd, device=device, dtype=dtype)
                     for _ in range(self.n_layers)]
        self.vbuf = [torch.zeros(bs, kvh, max_len, hd, device=device, dtype=dtype)
                     for _ in range(self.n_layers)]
        self.write_idx = None                       # [bs, t] long
        self.rows = None
        self.hi = None                              # attend only to buffer[:hi]                            # row indices [bs] long

    def update(self, key_states, value_states, layer_idx, *a, **kw):
        k, v = self.kbuf[layer_idx][self.rows], self.vbuf[layer_idx][self.rows]
        idx = self.write_idx[:, None, :, None].expand(-1, k.shape[1], -1, k.shape[3])
        k.scatter_(2, idx, key_states)
        v.scatter_(2, idx, value_states)
        self.kbuf[layer_idx][self.rows] = k
        self.vbuf[layer_idx][self.rows] = v
        hi = self.hi or k.shape[2]
        return k[:, :, :hi], v[:, :, :hi]

    def get_seq_length(self, layer_idx=0):
        return 0                                    # we pass explicit 4D masks + positions

    def get_max_cache_shape(self):
        return self.kbuf[0].shape[2]


class GraphedChildScorer:
    """CUDA-graph per-node child scoring: one replay instead of ~8 kernel launches.

    Captures exactly the eager math -- topk_8(log_softmax(U[d] + W2 @ W1[parent]))
    -- with static I/O buffers; tree construction order is unchanged (gate:
    token-exact vs eager). Enable with USE_CUDA_GRAPH=1.
    """

    def __init__(self, draft, k=8):
        self.k = k
        W1 = draft.markov_head.markov_w1.weight
        W2 = draft.markov_head.markov_w2.weight
        B, V = draft.block_size, W1.shape[0]
        self.static_U = torch.zeros(B, V, device="cuda", dtype=torch.float32)
        self.static_idx = torch.zeros(1, dtype=torch.long, device="cuda")
        self.static_d = torch.zeros(1, dtype=torch.long, device="cuda")

        def compute():
            w1row = W1.index_select(0, self.static_idx)[0]
            bias = (W2 @ w1row).float()
            urow = self.static_U.index_select(0, self.static_d)[0]
            lp = torch.log_softmax(urow + bias, dim=-1)
            return torch.topk(lp, self.k)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                compute()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out_vals, self.out_idx = compute()

    def new_round(self, U):
        self.static_U.copy_(U)

    def score(self, offset, parent_tok):
        """-> (values, indices) python lists for topk-8 of the conditional."""
        self.static_idx.fill_(parent_tok)
        self.static_d.fill_(offset)
        self.graph.replay()
        return self.out_vals.tolist(), self.out_idx.tolist()


def log_phat(logq):
    logit_q = logq - math.log1p(-min(math.exp(logq), 1 - 1e-9))
    z = PLATT_A * logit_q + PLATT_B
    return -math.log1p(math.exp(-z)) if z > -30 else z


@torch.no_grad()
def expand(draft, ctx, anchor_tok, mode, budget, theta, scorer=None):
    """Same expansion as dspark_value_tree.expand (markov / survival+theta)."""
    import heapq
    B = draft.block_size
    block_ids = torch.full((1, B), draft.mask_token_id, device=ctx.device, dtype=torch.long)
    block_ids[0, 0] = anchor_tok
    position_ids = torch.arange(0, ctx.shape[1] + B, device=ctx.device).unsqueeze(0)
    _, U = draft.backbone(block_ids, ctx, position_ids)
    U = U[0].float()
    W1 = draft.markov_head.markov_w1.weight
    W2 = draft.markov_head.markov_w2.weight
    branch_cap = 1 if mode == "chain" else 8
    if scorer is not None:
        scorer.new_round(U)
    tokens, parents, depths = [], [], []
    heap, tick = [], 0

    def push(parent_idx, parent_tok, prio, d):
        nonlocal tick
        if d > B:
            return
        if scorer is not None:
            vals, idxs = scorer.score(d - 1, parent_tok)
            pairs = list(zip(vals, idxs))[:branch_cap]
        else:
            lp = torch.log_softmax(U[d - 1] + (W2 @ W1[parent_tok]).float(), dim=-1)
            top = torch.topk(lp, branch_cap)
            pairs = list(zip(top.values.tolist(), top.indices.tolist()))
        for logq, tok in pairs:
            edge = logq if mode in ("chain", "markov") else log_phat(logq)
            heapq.heappush(heap, (-(prio + edge), tick, parent_idx, d, int(tok)))
            tick += 1

    push(-1, anchor_tok, 0.0, 1)
    while heap and len(tokens) < budget:
        neg, _, p_idx, d, tok = heapq.heappop(heap)
        if theta is not None and math.exp(-neg) < theta:
            break
        tokens.append(tok)
        parents.append(p_idx)
        depths.append(d)
        push(len(tokens) - 1, tok, -neg, d + 1)
    return tokens, parents, depths


class Request:
    def __init__(self, rid, input_ids, arrive_s):
        self.rid, self.input_ids, self.arrive = rid, input_ids, arrive_s
        self.row = None          # KV buffer row
        self.end = 0             # next free buffer offset (KV written so far)
        self.valid = None        # bool[MAX_LEN] committed-context positions
        self.logical = 0         # committed context length (positions for rope)
        self.anchor = None       # last committed token (next step's first input)
        self.feats = None        # [1, logical-1, feat] drafter context features
        self.done_tokens = 0
        self.finish = None


class Engine:
    def __init__(self, mode, temperature=0.0):
        self.mode = mode                            # nospec | chain7 | markov56 | adaptive
        self.temperature = float(temperature)
        self.tok = AutoTokenizer.from_pretrained(TARGET)
        self.target = AutoModelForCausalLM.from_pretrained(
            TARGET, torch_dtype=torch.bfloat16).to("cuda").eval()
        cfg = AutoConfig.from_pretrained(DSPARK_DIR)
        self.draft = DSparkDraft(cfg)
        self.draft.load_state_dict(load_file(f"{DSPARK_DIR}/model.safetensors"), strict=True)
        self.draft = self.draft.to(torch.bfloat16).to("cuda").eval()
        self.eos = self.tok.eos_token_id
        self.cache = RaggedCache(self.target.config, MAX_BS, MAX_LEN, "cuda", torch.bfloat16)
        self.free_rows = list(range(MAX_BS))
        self.t_draft = self.t_verify = 0.0
        self.scorer = GraphedChildScorer(self.draft) if os.environ.get("USE_CUDA_GRAPH") else None

    def cfg_for(self, bs):
        if self.mode == "chain7":
            return ("chain", 7, None)
        if self.mode == "markov56":
            return ("markov", 56, None)
        if self.mode == "adaptive":
            if os.environ.get("ADAPT_THETA"):
                return ("survival", 60, float(os.environ["ADAPT_THETA"]))
            if self.temperature > 0.0:
                # measured T>0 ladder: theta=0.05 wins at low bs; at high bs no
                # strict tree beats the chain (theta sweep 0.05/0.10/0.15/0.4 all
                # trail), so the policy degrades to its chain member exactly.
                return ("survival", 60, 0.05) if bs <= 8 else ("chain", 7, None)
            th = 0.02 if bs <= 4 else 0.05 if bs <= 8 else 0.15
            return ("survival", 60, th)
        raise ValueError(self.mode)

    def prefill(self, r):
        L = r.input_ids.shape[1]
        r.row = self.free_rows.pop()
        r.valid = torch.zeros(MAX_LEN, dtype=torch.bool, device="cuda")
        self.cache.rows = torch.tensor([r.row], device="cuda")
        self.cache.write_idx = torch.arange(L, device="cuda").unsqueeze(0)
        self.cache.hi = L
        mask = torch.full((1, 1, L, L), NEG, device="cuda", dtype=torch.bfloat16)
        causal = torch.arange(L, device="cuda")[None, :] <= torch.arange(L, device="cuda")[:, None]
        mask[0, 0][causal] = 0.0
        pos = torch.arange(L, device="cuda").unsqueeze(0)
        out = self.target(r.input_ids, past_key_values=self.cache, position_ids=pos,
                          attention_mask=mask, output_hidden_states=True, use_cache=True)
        r.valid[:L] = True
        r.end = L
        r.logical = L + 1
        r.anchor = int(out.logits[0, -1].argmax())
        r.feats = dspark_context_features(out, self.draft.target_layer_ids)  # [1, L, f]
        r.done_tokens = 1

    def release(self, r):
        self.free_rows.append(r.row)

    def step(self, active, now):
        bs = len(active)
        mode, budget, theta = (None, 0, None) if self.mode == "nospec" else self.cfg_for(bs)
        trees = []
        t0 = time.perf_counter()
        for r in active:
            if self.mode == "nospec":
                trees.append(([], [], [], None))
            elif self.temperature > 0.0:
                bc = 1 if mode == "chain" else 8
                tk, pa, de, vctx = TS.expand(
                    self.draft, r.feats, r.anchor, temperature=self.temperature,
                    theta=(theta or 0.0), node_cap=budget, branch_cap=bc)
                trees.append((tk, pa, de, vctx))
            else:
                tk, pa, de = expand(self.draft, r.feats, r.anchor, mode, budget, theta, self.scorer)
                trees.append((tk, pa, de, None))
        torch.cuda.synchronize()
        self.t_draft += time.perf_counter() - t0

        t0 = time.perf_counter()
        tmax = 1 + max(len(t[0]) for t in trees)     # [anchor] + tree, padded
        hi = max(r.end for r in active) + tmax
        tok_l, pos_l, anc_l = [], [], []
        for r, (toks, parents, depths, _v) in zip(active, trees):
            n = 1 + len(toks)
            row_tok = [r.anchor] + toks + [0] * (tmax - n)
            row_pos = [r.logical - 1] + [r.logical - 1 + d for d in depths] + [0] * (tmax - n)
            anc = [1 << 0]                                              # anchor: itself
            for j, p in enumerate(parents):
                anc.append(anc[p + 1] | (1 << (j + 1)))                 # ancestors bitmask
            anc += [0] * (tmax - n)
            tok_l.append(row_tok); pos_l.append(row_pos); anc_l.append(anc)
        step_tok = torch.tensor(tok_l, dtype=torch.long, device="cuda")
        pos = torch.tensor(pos_l, dtype=torch.long, device="cuda")
        anc_t = torch.tensor(anc_l, dtype=torch.int64, device="cuda")   # [bs, tmax]
        bits = (anc_t.unsqueeze(-1) >> torch.arange(tmax, device="cuda")) & 1   # [bs, tq, tk]
        mask = torch.full((bs, 1, tmax, hi), NEG, device="cuda", dtype=torch.bfloat16)
        valid_all = torch.stack([torch.nn.functional.pad(r.valid[:hi], (0, max(0, hi - r.valid[:hi].shape[0]))) for r in active])
        mask[:, 0, :, :] = torch.where(valid_all[:, None, :], 0.0, NEG).to(torch.bfloat16)
        for i, r in enumerate(active):                                  # tree block per row
            tree_m = torch.where(bits[i].bool(), 0.0, NEG).to(torch.bfloat16)
            mask[i, 0, :, r.end : r.end + tmax] = tree_m
            mask[i, 0, :, r.end + tmax : hi] = NEG
        n_real = torch.tensor([1 + len(t[0]) for t in trees], device="cuda")
        pad_row = torch.arange(tmax, device="cuda")[None, :] >= n_real[:, None]
        col0 = mask[:, 0, :, 0]
        mask[:, 0, :, 0] = torch.where(pad_row, torch.zeros_like(col0), col0)
        widx = torch.stack([torch.arange(r.end, r.end + tmax, device="cuda") for r in active])
        self.cache.rows = torch.tensor([r.row for r in active], device="cuda")
        self.cache.write_idx = widx
        self.cache.hi = hi
        out = self.target(step_tok, past_key_values=self.cache, position_ids=pos,
                          attention_mask=mask, output_hidden_states=True, use_cache=True)
        torch.cuda.synchronize()
        self.t_verify += time.perf_counter() - t0

        finished = []
        hs = dspark_context_features(out, self.draft.target_layer_ids)   # [bs, tmax, f]
        for i, (r, (toks, parents, depths, vctx)) in enumerate(zip(active, trees)):
            logits = out.logits[i].float()
            if self.mode == "nospec":
                if self.temperature > 0.0:
                    accepted = [sample_from_probs(probs_from_logits(logits[0], self.temperature))]
                    path = []
                else:
                    accepted, path = [int(logits[0].argmax())], []
            elif self.temperature > 0.0:
                accepted, path = TS._walk_recompute(
                    logits, toks, parents, depths, 1, self.temperature,
                    vctx["U"], r.anchor, self.draft)
            else:
                accepted, path = walk_tree(logits, toks, None, parents, 1, 0.0)
            stop = self.eos in accepted
            if stop:
                accepted = accepted[: accepted.index(self.eos) + 1]
            # commit: anchor + accepted-path tree nodes become valid context
            r.valid[r.end] = True
            for j in path:
                r.valid[r.end + 1 + j] = True
            feat_rows = [0] + [1 + j for j in path]
            r.feats = torch.cat([r.feats, hs[i:i+1, feat_rows]], dim=1)
            r.end += tmax
            r.logical += len(accepted)
            r.anchor = accepted[-1]
            r.done_tokens += len(accepted)
            if stop or r.done_tokens >= MAX_NEW or r.end + 64 >= MAX_LEN:
                r.finish = now()
                finished.append(r)
        return finished


@torch.no_grad()
def run(mode, n_requests=48, lam=None, seed=0, temperature=0.0, engine=None):
    """Closed pool of n_requests; lam=None -> all arrive at t=0 (saturation)."""
    import random
    rng = random.Random(seed)
    eng = engine or Engine(mode, temperature=temperature)
    eng.t_draft = eng.t_verify = 0.0
    torch.manual_seed(seed)
    prompts = []
    for name in ("gsm8k", "math500", "humaneval", "mbpp", "mt-bench", "alpaca"):
        with open(f"{DATA_DIR}/{name}.jsonl") as f:
            for i, line in enumerate(f):
                if i >= 8:
                    break
                prompts.append(json.loads(line)["turns"][0])
    reqs = []
    t_arr = 0.0
    for i in range(n_requests):
        text = eng.tok.apply_chat_template([{"role": "user", "content": prompts[i % len(prompts)]}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        ids = eng.tok(text, return_tensors="pt").input_ids[:, :512].to("cuda")
        reqs.append(Request(i, ids, t_arr))
        if lam:
            t_arr += rng.expovariate(lam)
    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0
    queue, active, done = list(reqs), [], []
    while queue or active:
        while queue and len(active) < MAX_BS and queue[0].arrive <= now():
            r = queue.pop(0)
            eng.prefill(r)
            active.append(r)
        if not active:
            time.sleep(0.001)
            continue
        for r in eng.step(active, now):
            active.remove(r)
            eng.release(r)
            done.append(r)
    wall = now()
    toks = sum(r.done_tokens for r in done)
    lat = sorted(r.finish - r.arrive for r in done)
    return {"mode": mode, "wall_s": wall, "goodput": toks / wall,
            "p50_s": lat[len(lat) // 2], "p99_s": lat[int(len(lat) * 0.99)],
            "t_draft_s": eng.t_draft, "t_verify_s": eng.t_verify, "n": len(done)}


if __name__ == "__main__":
    import sys
    mode = sys.argv[1]
    temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    r = run(mode, temperature=temp)
    print(json.dumps(r, indent=1))
