"""TreeSpark: calibrated-survival draft trees for semi-autoregressive drafters.

ONE method, no mode zoo. Per round: one drafter backbone forward; best-first
expansion where the only priority is calibrated path survival; a marginal-gain
threshold theta stops the round; one target forward verifies the tree
losslessly in either decoding mode.

    q(. | parent)  = softmax(U_d + B(parent))            Markov-head conditional
    p_hat(edge)    = sigmoid(a * logit(q) + b)           two-scalar calibration,
                                                          fit on ancestors-accepted
                                                          edges (labels are free
                                                          by-products of verification)
    survival(node) = prod p_hat(edges on root path)      = P(node is accepted)
    stop           when best candidate survival < theta  (theta = marginal price
                                                          of one verified token;
                                                          a scheduler sets it from
                                                          load, 0 at bs=1)

Token choice within a node is the drafter's own q (policy); survival governs
only where budget goes (value). Greedy (T=0) expands top-k children and
verifies by exact match; sampling (T>0) draws children without replacement
from the residual conditional and verifies with recursive rejection
sampling, recomputing each visited parent's proposals at verification time
instead of storing them -- both modes emit exactly the target distribution
(shape decisions never enter the acceptance rule).

Fixed budgets, chains, marginal scoring, and the log-q priority survive only
as ablations: chain = (node_cap=B, branch_cap=1, theta=0); fixed-N =
(node_cap=N, theta=0). The confidence head is not used (measured: no per-edge
signal beyond q).

All drafter/verification primitives are part of this package.
"""

import heapq
import math
import os

import torch

from treespark.dspark import dspark_context_features
from treespark.sampling import probs_from_logits, sample_from_probs
from treespark.tree_speculative import (
    build_tree_mask_and_positions,
    walk_tree,
)

# v2 calibration: fit on ancestors-accepted edges, Qwen3-4B greedy (test ECE 0.011).
# Per-scale refits (see experiments/supplemental_20260808) stay within a few
# percent; override per (model, temperature) when available.
PLATT_A, PLATT_B = 0.6684, -0.4587


def log_phat(logq, a=PLATT_A, b=PLATT_B):
    """log sigmoid(a * logit(q) + b) from natural-log q."""
    logit_q = logq - math.log1p(-min(math.exp(logq), 1 - 1e-9))
    z = a * logit_q + b
    return -math.log1p(math.exp(-z)) if z > -30 else z


@torch.no_grad()
def expand(draft, ctx, anchor_tok, *, temperature=0.0, theta=0.0,
           node_cap=64, branch_cap=8):
    """One backbone pass -> best-first tree under calibrated-survival priority.

    Returns (tokens, parents, depths, vctx); vctx is None at T=0, else
    {"U", "q_list"} where q_list stays None -- the verifier recomputes
    proposals from (U, draw order) -- unless TS_LEGACY_VERIFY=1 stores
    per-sibling snapshots for the equivalence gate.
    """
    B = draft.block_size
    device = ctx.device
    block_ids = torch.full((1, B), draft.mask_token_id, device=device, dtype=torch.long)
    block_ids[0, 0] = anchor_tok
    position_ids = torch.arange(0, ctx.shape[1] + B, device=device).unsqueeze(0)
    _, U = draft.backbone(block_ids, ctx, position_ids)
    U = U[0].float()
    W1 = draft.markov_head.markov_w1.weight
    W2 = draft.markov_head.markov_w2.weight

    def conditional(offset, parent_tok):
        return U[offset - 1] + (W2 @ W1[parent_tok]).float()

    tokens, parents, depths = [], [], []

    if temperature == 0.0:
        # Greedy: children are top-k of the conditional; survival is exact.
        heap, tick = [], 0                      # (-log_surv, tick, parent, depth, tok)

        def push_children(parent_idx, parent_tok, parent_logsurv, child_depth):
            nonlocal tick
            if child_depth > B:
                return
            lp = torch.log_softmax(conditional(child_depth, parent_tok), dim=-1)
            top = torch.topk(lp, branch_cap)
            for logq, tok in zip(top.values.tolist(), top.indices.tolist()):
                s = parent_logsurv + log_phat(logq)
                heapq.heappush(heap, (-s, tick, parent_idx, child_depth, int(tok)))
                tick += 1

        push_children(-1, anchor_tok, 0.0, 1)
        while heap and len(tokens) < node_cap:
            neg, _, p_idx, d, tok = heapq.heappop(heap)
            if theta > 0.0 and math.exp(-neg) < theta:
                break                            # marginal expected gain below price
            tokens.append(tok)
            parents.append(p_idx)
            depths.append(d)
            push_children(len(tokens) - 1, tok, -neg, d + 1)
        return tokens, parents, depths, None

    # Sampling: per-parent sibling draws are realized by ONE
    # torch.multinomial(replacement=False) call -- the exact sequential
    # without-replacement draw sequence (draw order = output order). All
    # per-parent GPU work (the softmaxes, the k draws, q0 top-9 for bounds,
    # value gathers) happens once, in one readback; slot gating then runs on
    # cached Python floats. Gates consult only pre-draw bounds and materialized
    # info -- a pre-drawn value is never read by its own admission decision, so
    # the filtration discipline (and Prop. 1) is untouched. The verifier
    # recomputes residuals from (U, parents, draw order) exactly as before.
    store = bool(os.environ.get("TS_LEGACY_VERIFY"))
    q_list = [] if store else None
    node_tok = {-1: anchor_tok}
    log_surv = {-1: 0.0}
    pdata, n_drawn = {}, {}
    heap, tick = [], 0

    def prep_parents(items):
        """ONE batched GPU visit for a group of parents; everything downstream is
        Python floats. torch.multinomial(..., replacement=False) realizes the
        exact sequential without-replacement draw sequence per row (draw order =
        output order). Unpopped parents prepped opportunistically are harmless:
        their draws are never read by any decision unless their slot is popped.
        At T=1 the sampling law equals q0, so one softmax serves both roles."""
        idxs = [i for i, _ in items]
        ptoks = torch.tensor([node_tok[i] for i, _ in items], device=U.device)
        drows = torch.tensor([d for _, d in items], device=U.device)
        bias = (draft.markov_head.markov_w2.weight @
                draft.markov_head.markov_w1.weight[ptoks].T).T.float()
        logits = U[drows] + bias                               # [m, V]
        q0 = torch.softmax(logits, dim=-1)
        qT = q0 if temperature == 1.0 else torch.softmax(logits / temperature, dim=-1)
        draws = torch.multinomial(qT, branch_cap, replacement=False)   # [m, k]
        top9 = torch.topk(q0, 9, dim=-1)
        pack = torch.cat([q0.gather(1, draws), top9.values], dim=1).tolist()
        dtoks = draws.tolist()
        t9tok = top9.indices.tolist()
        for r, pi in enumerate(idxs):
            pdata[pi] = {
                "toks": dtoks[r],
                "q0": pack[r][:branch_cap],
                "t9": list(zip(t9tok[r], pack[r][branch_cap:])),
                "drawn": set(),
            }

    def bound_edge(parent_idx):
        d = pdata[parent_idx]
        for t, v in d["t9"]:
            if t not in d["drawn"]:
                return v
        return 0.0

    def push_slot(parent_idx, parent_depth):
        nonlocal tick
        if parent_depth + 1 > B:
            return
        if parent_idx in pdata:
            top = bound_edge(parent_idx)
            b = log_surv[parent_idx] + (log_phat(math.log(max(top, 1e-12))) if top > 0 else -1e9)
        else:
            b = log_surv[parent_idx]            # admissible: child survival < parent survival
        heapq.heappush(heap, (-b, tick, parent_idx, parent_depth))
        tick += 1

    push_slot(-1, 0)
    while heap and len(tokens) < node_cap:
        neg, _, p_idx, p_depth = heapq.heappop(heap)
        if theta > 0.0 and math.exp(-neg) < theta:
            break                               # admissible bound < theta => true < theta
        if p_idx not in pdata:                  # lazy: prep this parent AND every other
            pending = {p_idx: p_depth}          # unprepped parent queued in the heap,
            for _nb, _t, qi, qd in heap:        # in one batched GPU visit
                if qi not in pdata and qi not in pending:
                    pending[qi] = qd
            prep_parents(list(pending.items()))
            push_slot(p_idx, p_depth)
            continue
        d = pdata[p_idx]
        j = n_drawn.get(p_idx, 0)
        if j >= len(d["toks"]):
            continue
        tok = d["toks"][j]
        child_depth = p_depth + 1
        tokens.append(tok)
        parents.append(p_idx)
        depths.append(child_depth)
        idx = len(tokens) - 1
        node_tok[idx] = tok
        log_surv[idx] = log_surv[p_idx] + log_phat(math.log(max(d["q0"][j], 1e-12)))
        d["drawn"].add(tok)
        n_drawn[p_idx] = j + 1
        push_slot(idx, child_depth)
        if n_drawn[p_idx] < branch_cap:
            push_slot(p_idx, p_depth)
    if store:
        # legacy testing path: rebuild the residual proposals the draws came from
        kids = {}
        for i, p in enumerate(parents):
            kids.setdefault(p, []).append(i)
        q_list = [None] * len(tokens)
        for p, cs in kids.items():
            depth0 = 1 if p == -1 else depths[p] + 1
            qm = torch.softmax(conditional(depth0, node_tok[p]) / temperature, dim=-1)
            for c in cs:
                q_list[c] = (qm / qm.sum()).clone()
                qm[tokens[c]] = 0.0
    return tokens, parents, depths, {"U": U, "q_list": q_list}


@torch.no_grad()
def _walk_recompute(all_logits, tokens, parents, depths, prefix_len, temperature,
                    U, anchor_tok, draft):
    """Rejection walk with proposals RECOMPUTED on demand instead of stored.

    Each visited parent's temperature-scaled conditional is rebuilt with the
    same fp ops (and order) the expansion used, and the without-replacement
    residual is replayed in draw order (= materialization order), so every
    proposal is bit-identical to the one the sibling was sampled from --
    losslessness is untouched. Only parents on the walk are touched (~tau
    per round, not N), and nothing is cloned at expansion time.
    """
    W1 = draft.markov_head.markov_w1.weight
    W2 = draft.markov_head.markov_w2.weight
    kids = {}
    for i, p in enumerate(parents):
        kids.setdefault(p, []).append(i)        # draw order == index order

    accepted, path = [], []
    cur_node, cur_row = -1, prefix_len - 1
    while True:
        p_cur = probs_from_logits(all_logits[cur_row], temperature)
        children = kids.get(cur_node, [])
        if not children:
            accepted.append(sample_from_probs(p_cur))
            break
        child_depth = 1 if cur_node == -1 else depths[cur_node] + 1
        parent_tok = anchor_tok if cur_node == -1 else tokens[cur_node]
        logits = U[child_depth - 1] + (W2 @ W1[parent_tok]).float()
        qm = torch.softmax(logits / temperature, dim=-1)
        matched = None
        for child in children:
            x = tokens[child]
            mass = float(qm.sum())
            qx = float(qm[x]) / mass if mass > 0 else 0.0
            if qx > 0.0 and torch.rand(()) < p_cur[x] / qx:
                matched = child
                accepted.append(x)
                break
            proposal = qm / mass                # full residual, rejected tries only
            p_cur = torch.clamp(p_cur - proposal, min=0.0)
            s = float(p_cur.sum())
            if s <= 0.0:                        # fp-degenerate residual (prob-0 event)
                p_cur = proposal
            else:
                p_cur = p_cur / s
            qm[x] = 0.0
        if matched is None:
            accepted.append(sample_from_probs(p_cur))
            break
        path.append(matched)
        cur_node, cur_row = matched, prefix_len + matched
    return accepted, path


@torch.no_grad()
def generate(target, draft, input_ids, max_new_tokens, *, temperature=0.0,
             theta=0.0, node_cap=64, branch_cap=8, eos_id=None):
    """Cache-free reference decode loop; lossless at any (temperature, theta)."""
    ids = input_ids.clone()
    device = ids.device
    out = target(ids, output_hidden_states=True)
    if temperature == 0.0:
        anchor = int(out.logits[0, -1].argmax())
    else:
        anchor = sample_from_probs(probs_from_logits(out.logits[0, -1].float(), temperature))
    ids = torch.cat([ids, torch.tensor([[anchor]], device=device, dtype=ids.dtype)], dim=-1)
    feats = dspark_context_features(out, draft.target_layer_ids)

    target_forward, num_generated, verified_total = 0, 1, 0
    tree_sizes = []
    while num_generated < max_new_tokens:
        n = ids.shape[1]
        anchor_tok = ids[0, -1].item()
        tokens, parents, depths, vctx = expand(
            draft, feats, anchor_tok, temperature=temperature, theta=theta,
            node_cap=node_cap, branch_cap=branch_cap)
        verified_total += len(tokens)
        tree_sizes.append(len(tokens))
        mask, positions = build_tree_mask_and_positions(n, parents, depths, device, target.dtype)
        cand = torch.cat([ids, torch.tensor([tokens], device=device, dtype=ids.dtype)], dim=-1)
        out = target(cand, attention_mask=mask, position_ids=positions, output_hidden_states=True)
        target_forward += 1

        if temperature == 0.0:
            accepted, path = walk_tree(out.logits[0].float(), tokens, None, parents, n, 0.0)
        elif vctx.get("q_list") is not None:    # legacy store-path (testing only)
            accepted, path = walk_tree(out.logits[0].float(), tokens, vctx["q_list"], parents, n, temperature)
        else:
            accepted, path = _walk_recompute(out.logits[0].float(), tokens, parents,
                                             depths, n, temperature, vctx["U"],
                                             anchor_tok, draft)
        stop = eos_id is not None and eos_id in accepted
        if stop:
            accepted = accepted[: accepted.index(eos_id) + 1]
        ids = torch.cat([ids, torch.tensor([accepted], device=device, dtype=ids.dtype)], dim=-1)
        num_generated += len(accepted)
        if stop:
            break
        hs = dspark_context_features(out, draft.target_layer_ids)
        keep = list(range(n)) + [n + j for j in path]
        feats = hs[:, keep][:, : ids.shape[1] - 1]

    return ids, {
        "target_forward": target_forward,
        "num_generated_tokens": num_generated,
        "tokens_per_forward": num_generated / max(1, target_forward),
        "avg_verify_len": verified_total / max(1, target_forward),
        "tree_sizes": tree_sizes,
    }
