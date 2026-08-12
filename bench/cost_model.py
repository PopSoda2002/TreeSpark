"""Measured decode-step cost model T(bs, k): batch size x verify-tokens-per-request.

Setup mirrors one speculative decoding round under batching: every request holds a
512-token prefix in KV cache; the step forward processes k draft tokens per request
(tree vs chain mask shape does not change token count, which dominates cost).

For each (bs, k): prefill once (use_cache), then time the k-token forward with the
cached prefix, cropping the cache back between reps. Median of 15 reps, 3 warmup.

Run: CUDA_VISIBLE_DEVICES=<g> python serving_cost_model.py
Output: results/serving_cost_model.json  {bs: {k: ms}}
"""

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM

TARGET = "Qwen/Qwen3-4B"
PREFIX_LEN = 512
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
KS = [1, 7, 13, 23, 28, 56]
REPS, WARMUP = 15, 3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "serving_cost_model.json")


@torch.no_grad()
def main():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TARGET, torch_dtype=torch.bfloat16).to("cuda").eval()
    vocab = model.config.vocab_size
    results = {}
    for bs in BATCH_SIZES:
        prefix = torch.randint(0, vocab, (bs, PREFIX_LEN), device="cuda")
        out = model(prefix, use_cache=True)
        cache = out.past_key_values
        results[bs] = {}
        for k in KS:
            step = torch.randint(0, vocab, (bs, k), device="cuda")
            pos = torch.arange(PREFIX_LEN, PREFIX_LEN + k, device="cuda").unsqueeze(0).expand(bs, -1)
            times = []
            for i in range(WARMUP + REPS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(step, past_key_values=cache, position_ids=pos, use_cache=True)
                torch.cuda.synchronize()
                if i >= WARMUP:
                    times.append((time.perf_counter() - t0) * 1000)
                cache.crop(PREFIX_LEN)
            results[bs][k] = sorted(times)[len(times) // 2]
        del cache, out
        torch.cuda.empty_cache()
        print(f"bs={bs:<4} " + "  ".join(f"k={k}:{results[bs][k]:6.1f}ms" for k in KS), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
