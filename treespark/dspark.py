"""DSpark: semi-autoregressive drafting (DeepSeek, alphaxiv 2026.dspark).

Sits between DFlash (fully parallel, suffers suffix decay) and EAGLE (fully
autoregressive, serial bottleneck):

  1. a DFlash-style parallel BACKBONE runs one forward over the block and emits
     base logits U_1..U_g for every position at once (fast, but position-independent)
  2. a lightweight MARKOV head adds intra-block dependency via a LOW-RANK transition
     bias B_k(x_{k-1}, v) = <W1[x_{k-1}], W2[v]>, so token k's distribution is
     softmax(U_k + B_k(x_{k-1})). Sampling stays sequential (cheap: a lookup + matvec)
     while the heavy forward is parallel -> "semi-autoregressive"
  3. a CONFIDENCE head predicts per-position accept probability c_k for load-aware
     verification scheduling (the production cousin of EAGLE-2's path score)

Checkpoint: deepseek-ai/dspark_qwen3_4b_block7 (block 7, target Qwen3-4B, 5 backbone
layers, markov rank 256, taps [1,9,17,25,33]); ships its own embed_tokens + lm_head."""

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding, repeat_kv, rotate_half


class DSparkAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, hd = config.hidden_size, config.head_dim
        self.head_dim = hd
        self.n_rep = config.num_attention_heads // config.num_key_value_heads
        self.q_proj = nn.Linear(h, config.num_attention_heads * hd, bias=False)
        self.k_proj = nn.Linear(h, config.num_key_value_heads * hd, bias=False)
        self.v_proj = nn.Linear(h, config.num_key_value_heads * hd, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * hd, h, bias=False)
        self.q_norm = Qwen3RMSNorm(hd, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(hd, eps=config.rms_norm_eps)

    def forward(self, block_hidden, ctx, pos_emb):
        B, S, _ = block_hidden.shape
        C = ctx.shape[1]
        q = self.q_norm(self.q_proj(block_hidden).view(B, S, -1, self.head_dim)).transpose(1, 2)
        k = torch.cat([self.k_proj(ctx), self.k_proj(block_hidden)], dim=1)
        v = torch.cat([self.v_proj(ctx), self.v_proj(block_hidden)], dim=1)
        k = self.k_norm(k.view(B, C + S, -1, self.head_dim)).transpose(1, 2)
        v = v.view(B, C + S, -1, self.head_dim).transpose(1, 2)
        cos, sin = pos_emb
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q = (q * cos[..., -S:, :]) + (rotate_half(q) * sin[..., -S:, :])
        k = (k * cos) + (rotate_half(k) * sin)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)   # bidirectional
        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1))


class DSparkLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = DSparkAttention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, ctx, pos_emb):
        x = x + self.self_attn(self.input_layernorm(x), ctx, pos_emb)
        return x + self.mlp(self.post_attention_layernorm(x))


class MarkovHead(nn.Module):
    """Low-rank transition bias: B_k(x_{k-1}, v) = <W1[x_{k-1}], W2[v]>.
    Full V x V would be 152k^2; rank-256 factorization makes it a lookup + matvec."""
    def __init__(self, config):
        super().__init__()
        rank = config.markov_rank or 1                              # rank 0 = DFlash-style ckpt
        self.markov_w1 = nn.Linear(rank, config.vocab_size, bias=False)   # weight [V, r]: row v = W1[v]
        self.markov_w2 = nn.Linear(rank, config.vocab_size, bias=False)   # weight [V, r]: row v = W2[v]

    def bias_for(self, prev_token):
        """Bias added to every candidate's logit given the previously sampled token."""
        u = self.markov_w1.weight[prev_token]          # [r]
        return self.markov_w2.weight @ u               # [V]  = <W2[v], W1[prev]> for all v


class ConfidenceHead(nn.Module):
    """c_k = sigmoid(w . [h_k ; W1[x_{k-1}]]) -> per-position accept probability."""
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size + (config.markov_rank or 1), 1)

    def forward(self, h_k, w1_prev):
        return torch.sigmoid(self.proj(torch.cat([h_k, w1_prev], dim=-1)))


class DSparkDraft(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, v = config.hidden_size, config.vocab_size
        self.embed_tokens = nn.Embedding(v, h)
        self.fc = nn.Linear(len(config.target_layer_ids) * h, h, bias=False)
        self.hidden_norm = Qwen3RMSNorm(h, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(DSparkLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = Qwen3RMSNorm(h, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(h, v, bias=False)
        self.markov_head = MarkovHead(config)
        self.confidence_head = ConfidenceHead(config)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.target_layer_ids = config.target_layer_ids
        self.mask_token_id = config.mask_token_id
        self.block_size = config.block_size

    def backbone(self, block_ids, ctx_feats, position_ids):
        """One parallel forward -> per-position hidden states h and base logits U."""
        ctx = self.hidden_norm(self.fc(ctx_feats))
        block_embeds = self.embed_tokens(block_ids)
        pos_emb = self.rotary_emb(block_embeds, position_ids)
        h = block_embeds
        for layer in self.layers:
            h = layer(h, ctx, pos_emb)
        h = self.norm(h)
        return h, self.lm_head(h)


def dspark_context_features(outputs, layer_ids):
    hs = outputs.hidden_states
    return torch.cat([hs[i + 1] for i in layer_ids], dim=-1)


@torch.no_grad()
def dspark_draft_block(draft, ctx, anchor_tok, use_markov=True):
    """Semi-autoregressive draft: ONE parallel backbone pass over the block, then
    SEQUENTIAL sampling where each token adds a low-rank markov bias from the previous
    one. The heavy forward is parallel; the serial part is just a lookup + matvec.

    Block semantics are LM-SHIFTED (matches DeepSpec's official eval): row k of the
    block (row 0 = anchor embedding, rows 1.. = mask embeddings) predicts the draft
    at offset k+1 after the anchor, so one block yields block_size drafts. Probed
    empirically on dspark_qwen3_4b: shifted 5.89 vs in-place 1.67 accepted/round.

    use_markov=False is the ablation: the backbone's base logits U alone (pure
    parallel). DSpark's backbone is co-trained WITH the markov head, so standalone it
    is much weaker than DFlash's -- the markov correction is not optional here."""
    B = draft.block_size
    device = ctx.device
    block_ids = torch.full((1, B), draft.mask_token_id, device=device, dtype=torch.long)
    block_ids[0, 0] = anchor_tok
    position_ids = torch.arange(0, ctx.shape[1] + B, device=device).unsqueeze(0)

    h, U = draft.backbone(block_ids, ctx, position_ids)   # [1, B, h], [1, B, V]
    U = U[0].float()

    drafts, confs, prev = [], [], anchor_tok
    for k in range(B):                                    # row k -> draft at offset k+1
        w1_prev = draft.markov_head.markov_w1.weight[prev]                 # [r]
        logit = U[k] + (draft.markov_head.markov_w2.weight @ w1_prev.to(draft.markov_head.markov_w2.weight.dtype)).float() \
            if use_markov else U[k]
        tok = int(logit.argmax())
        c_k = float(draft.confidence_head(h[0, k], w1_prev))              # accept probability
        drafts.append(tok)
        confs.append(c_k)
        prev = tok
    return drafts, confs


@torch.no_grad()
def dspark_generate(target, draft, input_ids, max_new_tokens, eos_id=None, use_markov=True, conf_threshold=0.0):
    """v1: greedy, cache-free, official match-to-posterior acceptance.

    conf_threshold > 0 enables confidence-scheduled verification (single-request
    version): keep only the draft prefix whose survival probability prod(c_k) stays
    above the threshold, then verify just that prefix. High threshold ~ heavy load
    (aggressively drop the low-survival tail); 0 ~ light load (verify the whole block).
    The full DSpark scheduler replaces the fixed threshold with tau*SPS(B) optimization."""
    B = draft.block_size
    ids = input_ids.clone()
    device = ids.device

    out = target(ids, output_hidden_states=True)
    anchor = int(out.logits[0, -1].argmax())
    ids = torch.cat([ids, torch.tensor([[anchor]], device=device, dtype=ids.dtype)], dim=-1)
    feats = dspark_context_features(out, draft.target_layer_ids)

    target_forward, num_generated, verified_total = 0, 1, 0
    while num_generated < max_new_tokens:
        n = ids.shape[1]
        drafts, confs = dspark_draft_block(draft, feats, ids[0, -1].item(), use_markov=use_markov)

        # confidence scheduling: truncate where survival probability drops below threshold
        if conf_threshold > 0.0:
            survival = torch.cumprod(torch.tensor(confs), dim=0)
            keep = int((survival >= conf_threshold).sum())
            drafts = drafts[: max(1, keep)]
        L = len(drafts)
        verified_total += L

        cand = torch.cat([ids, torch.tensor([drafts], device=device, dtype=ids.dtype)], dim=-1)
        out = target(cand, output_hidden_states=True)
        target_forward += 1
        posterior = out.logits[0, n - 1 : n - 1 + L + 1].argmax(dim=-1)

        acc = 0
        while acc < L and drafts[acc] == int(posterior[acc]):
            acc += 1
        new = drafts[:acc] + [int(posterior[acc])]
        stop = eos_id is not None and eos_id in new
        if stop:
            new = new[: new.index(eos_id) + 1]
        ids = torch.cat([ids, torch.tensor([new], device=device, dtype=ids.dtype)], dim=-1)
        num_generated += len(new)
        if stop:
            break
        feats = dspark_context_features(out, draft.target_layer_ids)[:, : ids.shape[1] - 1]
    stats = {
        "target_forward": target_forward,
        "num_generated_tokens": num_generated,
        "tokens_per_forward": num_generated / max(1, target_forward),
        "avg_verify_len": verified_total / max(1, target_forward),
    }
    return ids, stats


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from treespark.baseline import baseline_generate, build_prompt

    TARGET = "Qwen/Qwen3-4B"
    REPO = "deepseek-ai/dspark_qwen3_4b_block7"

    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    target = AutoModelForCausalLM.from_pretrained(TARGET, torch_dtype=torch.bfloat16).to("cuda").eval()
    config = AutoConfig.from_pretrained(REPO)
    draft = DSparkDraft(config)
    draft.load_state_dict(load_file(hf_hub_download(REPO, "model.safetensors")), strict=True)
    draft = draft.to(torch.bfloat16).to("cuda").eval()
    print(f"loaded OK, block_size {config.block_size}, markov_rank {config.markov_rank}")

    eos_id = tokenizer.eos_token_id
    input_ids = build_prompt(tokenizer, "Explain speculative decoding in three sentences.", "cuda")
    prompt_len = input_ids.shape[1]

    base_ids, base_stats = baseline_generate(target, input_ids, 120, temperature=0.0, eos_id=eos_id)
    ds_ids, s = dspark_generate(target, draft, input_ids, 120, eos_id=eos_id)
    nb = base_ids.shape[1]
    exact = ds_ids.shape[1] >= nb and torch.equal(base_ids, ds_ids[:, :nb])
    print(f"\nexact match (greedy): {exact}")
    print(f"baseline: {base_stats['target_forward']} target forwards")
    print(f"dspark:   tau {s['tokens_per_forward']:.2f}, {s['target_forward']} target forwards")

    # markov ablation: base logits only (backbone is co-trained to be corrected)
    _, s_nm = dspark_generate(target, draft, input_ids, 120, eos_id=eos_id, use_markov=False)
    print(f"markov OFF: tau {s_nm['tokens_per_forward']:.2f}")

    # S4: confidence-scheduled verification — sweep the survival-probability threshold.
    # higher threshold ~ heavier load: trade a little tau for a shorter verify length
    # (fewer target-batch slots spent on doomed tail drafts)
    print("\nconfidence-scheduled verification (single-request):")
    print(f"{'threshold':>10} {'tau':>6} {'avg_verify_len':>15}")
    for th in [0.0, 0.3, 0.5, 0.7, 0.9]:
        _, sc = dspark_generate(target, draft, input_ids, 120, eos_id=eos_id, conf_threshold=th)
        print(f"{th:>10.1f} {sc['tokens_per_forward']:>6.2f} {sc['avg_verify_len']:>15.2f}")
