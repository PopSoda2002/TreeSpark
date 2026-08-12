import torch

from treespark.sampling import probs_from_logits, sample_from_probs

def build_tree_mask_and_positions(prefix_len, parents, depths, device, dtype):
    total_len = prefix_len + len(parents)
    positions = [i for i in range(prefix_len)]
    for i in range(prefix_len, total_len):
        positions.append(prefix_len + depths[i - prefix_len] - 1)
    positions = torch.tensor(positions, device=device, dtype=torch.long).unsqueeze(0)
    visible = torch.tril(torch.ones((total_len, total_len), device=device, dtype=dtype))
    for i in range(prefix_len, total_len):
        visible[i, :] = 0
        visible[i, i] = 1
        visible[i, :prefix_len] = 1
        current = i - prefix_len
        while parents[current] != -1:
            visible[i, prefix_len + parents[current]] = 1
            current = parents[current]
    mask = torch.full((total_len, total_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask[visible.bool()] = 0
    mask = mask.reshape(1, 1, total_len, total_len)
    return mask, positions

# BFS to build the tree structure
def build_tree_structure(branching, depth):
    parents, depths = [], []
    cur_level = [-1]
    for d in range(1, depth + 1):
        next_level = []
        for node in cur_level:
            for k in range(branching):
                parents.append(node)
                depths.append(d)
                next_level.append(len(parents) - 1)
        cur_level = next_level
    return parents, depths

def pick_candidates(logits, branching, temperature):
    if temperature == 0.0:
        return torch.topk(logits, branching).indices.tolist()
    probs = probs_from_logits(logits, temperature)
    return [sample_from_probs(probs) for _ in range(branching)]

@torch.no_grad()
def draft_tree(draft_model, ids, branching, depth, temperature):
    parents, depths = build_tree_structure(branching, depth)
    tokens = [None] * len(parents)
    q_list = [None] * len(parents)
    logits = draft_model(ids).logits[0, -1]
    probs = probs_from_logits(logits, temperature)
    candidates = pick_candidates(logits, branching, temperature)
    for i in range(len(candidates)):
        q_list[i] = probs
        tokens[i] = candidates[i]
    if depth == 1:
        return tokens, q_list, parents, depths
    prefix_len = ids.shape[1]
    mask_full, positions_full = build_tree_mask_and_positions(prefix_len, parents, depths, ids.device, torch.float16)
    prev_level = [i for i in range(branching)]
    for d in range(2, depth + 1):
        m = depths.index(d)
        S = prefix_len + m
        candidates = torch.cat([ids, torch.tensor([tokens[:m]], device=ids.device, dtype=ids.dtype)], dim=-1)
        all_logits = draft_model(candidates, attention_mask=mask_full[:, :, :S, :S], position_ids=positions_full[:,:S]).logits[0]
        new_level = []
        for j in prev_level:
            probs = probs_from_logits(all_logits[prefix_len + j], temperature)
            next_tokens = pick_candidates(all_logits[prefix_len + j], branching, temperature)
            children = [i for i, p in enumerate(parents) if p == j]
            for child, token in zip(children, next_tokens):
                tokens[child] = token
                q_list[child] = probs
            new_level.extend(children)
        prev_level = new_level
    return tokens, q_list, parents, depths

def walk_tree(all_logits, tokens, q_list, parents, prefix_len, temperature):
    """Walk the verified tree from the prefix tail: emit the longest accepted
    path plus one correction/bonus token. Returns (accepted, path) where path
    holds the accepted nodes' tree indices."""
    accepted, path = [], []
    cur_node, cur_row = -1, prefix_len - 1
    while True:
        cur_probs = probs_from_logits(all_logits[cur_row], temperature)
        children = [i for i, p in enumerate(parents) if p == cur_node]
        if children == []:
            accepted.append(sample_from_probs(cur_probs))
            break
        if temperature == 0.0:
            token_star = int(cur_probs.argmax())
            accepted.append(token_star)
            matched = None
            for child in children:
                if tokens[child] == token_star:
                    matched = child
                    break
            if matched is None:
                break
        else:
            p_cur = cur_probs
            matched = None
            for child in children:
                x, q = tokens[child], q_list[child]
                if torch.rand(()) < p_cur[x] / q[x]:
                    matched = child
                    accepted.append(x)
                    break
                p_cur = torch.clamp(p_cur - q, min=0.0)
                p_cur = p_cur / p_cur.sum()
            if matched is None:
                accepted.append(sample_from_probs(p_cur))
                break
        path.append(matched)
        cur_node, cur_row = matched, prefix_len + matched
    return accepted, path

@torch.no_grad()
def verify_tree(target_model, ids, tokens, q_list, parents, depths, temperature):
    prefix_len = ids.shape[1]
    V = q_list[0].shape[0]
    mask_full, positions_full = build_tree_mask_and_positions(prefix_len, parents, depths, ids.device, torch.float16)
    candidates = torch.cat([ids, torch.tensor([tokens], device=ids.device, dtype=ids.dtype)], dim=-1)
    all_logits = target_model(candidates, attention_mask=mask_full, position_ids=positions_full).logits[0]
    accepted, _ = walk_tree(all_logits[:, :V], tokens, q_list, parents, prefix_len, temperature)
    return accepted, len(accepted) - 1

@torch.no_grad()
def tree_speculative_generate(target, draft, input_ids, max_new_tokens, branching, depth, temperature, eos_id=None):
    ids = input_ids.clone()
    draft_forward, target_forward, num_generated_tokens, accept_tokens = 0, 0, 0, 0
    while num_generated_tokens < max_new_tokens:
        L = ids.shape[1]
        # Draft phase
        draft_ids, q_list, parents, depths = draft_tree(draft, ids, branching, depth, temperature)
        draft_forward += depth
        # Verify phase
        accepted_ids, _ = verify_tree(target, ids, draft_ids, q_list, parents, depths, temperature)
        target_forward += 1
        accept_tokens += len(accepted_ids) - 1
        stop = eos_id is not None and eos_id in accepted_ids
        if stop:
            accepted_ids = accepted_ids[:accepted_ids.index(eos_id) + 1]
        # Extend phase
        ids = torch.cat([ids, torch.tensor([accepted_ids], device=ids.device, dtype=ids.dtype)], dim=-1)
        num_generated_tokens += len(accepted_ids)
        if stop:
            break
    stats = {
        "draft_forward": draft_forward,
        "target_forward": target_forward,
        "num_generated_tokens": num_generated_tokens,
        "accept_tokens": accept_tokens,
    }
    return ids, stats

if __name__ == "__main__":
    import time
    from transformers import AutoTokenizer
    from treespark.baseline import (
        DRAFT_MODEL,
        TARGET_MODEL,
        TARGET_DEVICE,
        baseline_generate,
        build_prompt,
        load_model,
    )

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    target = load_model(TARGET_MODEL, TARGET_DEVICE)
    draft = load_model(DRAFT_MODEL, TARGET_DEVICE)
    eos_id = tokenizer.eos_token_id
    input_ids = build_prompt(tokenizer, "Explain speculative decoding in three sentences.", TARGET_DEVICE)
    prompt_len = input_ids.shape[1]

    t0 = time.time()
    base_ids, base_stats = baseline_generate(target, input_ids, 100, temperature=0.0, eos_id=eos_id)
    t_base = time.time() - t0

    t0 = time.time()
    tree_ids, s = tree_speculative_generate(
        target, draft, input_ids, 100, branching=2, depth=3, temperature=0.0, eos_id=eos_id
    )
    t_tree = time.time() - t0

    print("=== tree speculative ===")
    print(tokenizer.decode(tree_ids[0, prompt_len:], skip_special_tokens=True))
    print()
    print(f"exact match (greedy): {torch.equal(base_ids, tree_ids)}")
    print(f"baseline: {base_stats['target_forward']} target forwards, {t_base:.1f}s")
    print(f"tree:     {s['target_forward']} target forwards, {s['draft_forward']} draft forwards, {t_tree:.1f}s")
    print(f"tokens per target forward: {s['num_generated_tokens'] / s['target_forward']:.2f} "
          f"(linear spec at gamma=3 measured ~1.88)")

    # sampling-mode smoke test
    tree_ids, s = tree_speculative_generate(
        target, draft, input_ids, 60, branching=2, depth=3, temperature=0.7, eos_id=eos_id
    )
    print()
    print("=== T=0.7 smoke test ===")
    print(tokenizer.decode(tree_ids[0, prompt_len:], skip_special_tokens=True))
    print(f"tokens per target forward: {s['num_generated_tokens'] / s['target_forward']:.2f}")
