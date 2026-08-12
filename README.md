# TreeSpark

Official implementation of *TreeSpark: Calibrated, Load-Adaptive Draft Trees
for Semi-Autoregressive Speculative Decoding*.

One calibrated acceptance estimate — a two-parameter map fitted on the
drafter's Markov conditional — governs the draft tree at every scale: which
node to expand next, when each round's tree stops growing (threshold `theta`),
and how much speculation the current serving load supports. Siblings are
sampled without replacement with matching residuals in recursive rejection,
so decoding is lossless at any temperature.

## Install

```bash
pip install -e .
```

## Quickstart

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from treespark import DSparkDraft, generate
from treespark.baseline import build_prompt

TARGET = "Qwen/Qwen3-4B"
DRAFTER = "deepseek-ai/dspark_qwen3_4b_block7"

tokenizer = AutoTokenizer.from_pretrained(TARGET)
target = AutoModelForCausalLM.from_pretrained(
    TARGET, torch_dtype=torch.bfloat16).to("cuda").eval()

config = AutoConfig.from_pretrained(DRAFTER)
draft = DSparkDraft(config)
draft.load_state_dict(load_file(hf_hub_download(DRAFTER, "model.safetensors")), strict=True)
draft = draft.to(torch.bfloat16).to("cuda").eval()

input_ids = build_prompt(tokenizer, "Explain speculative decoding in three sentences.", "cuda")
ids, stats = generate(target, draft, input_ids, 200, temperature=0.0, theta=0.02)
print(tokenizer.decode(ids[0, input_ids.shape[1]:], skip_special_tokens=True))
print(stats)
```

`generate(target, draft, input_ids, max_new_tokens, *, temperature=0.0,
theta=0.0, node_cap=64, branch_cap=8, eos_id=None)` is the reference decode
loop; `expand(...)` builds one round's tree and is the piece to embed in a
serving engine. Lower `theta` buys bigger trees (more speculation), higher
`theta` shrinks them toward the chain — a serving scheduler can set it per
round from load.

## Serving: the engine and load-aware theta

`treespark/engine.py` is a continuous-batching engine that runs adaptive
trees under load: a per-row ragged KV cache (no compaction), ancestor-bitmask
tree attention masks, an iteration-level scheduler, and an optional
CUDA-graphed child scorer (`USE_CUDA_GRAPH=1`).

`theta` is the knob that makes trees load-aware: it is the price of one
verified node, so the engine re-reads it every round from the current batch
size. The shipped policy is a measured *ladder* `theta*(bs, T)` — for each
(batch size, temperature) cell, certify which theta (or the plain chain) wins
wall-clock on your deployment, then interpolate. At low batch, verification
is nearly free and small theta (big trees) wins; as the batch grows the
forward becomes compute-bound and the ladder shrinks trees toward the chain,
which it recovers exactly as its `theta -> 1` member. Override the ladder
with `ADAPT_THETA=<x>` to probe a fixed theta.

Rebuild the ladder for a new GPU or model pair with the two measurement
tools below — it is a per-deployment object, not a constant.

## Measurement

```
bench/cost_model.py        measures the verification step cost L(bs, k) on
                           your GPU -> results/serving_cost_model.json
bench/wallclock_bench.py   wall-clock certification: both engines prebuilt on
                           the SAME GPU, arms interleaved rep by rep, warm-up
                           rep discarded, verdicts from paired diffs
bench/serving_sim.py       continuous-batching simulation (Poisson arrivals,
                           iteration scheduler) driven by the measured
                           L(bs, k) and real per-round traces
```

Wall-clock claims should come from `wallclock_bench.py` pairs only; cross-GPU or
sequential comparisons on shared boxes carry tenant noise. Inputs for
`serving_sim.py` (measured costs, round traces) ship in the release archive.

## Layout

```
treespark/
  tree.py              the method: calibrated best-first expansion, theta stop,
                       without-replacement sampling + recompute verification
  engine.py            continuous-batching serving engine + the theta ladder
  dspark.py            the semi-autoregressive drafter (backbone + Markov head)
  tree_speculative.py  tree attention mask + verification walk
  sampling.py          temperature sampling helpers
  baseline.py          target-only decode loop (reference/baseline)
bench/
  cost_model.py        L(bs, k) measurement
  wallclock_bench.py   interleaved A/B wall-clock certification
  serving_sim.py       load simulation from measured costs
```

## Paper

*TreeSpark: Calibrated, Load-Adaptive Draft Trees for Semi-Autoregressive
Speculative Decoding.* Raw experiment artifacts and the reproduction manual
are attached to this repository's
[releases](https://github.com/PopSoda2002/TreeSpark/releases).

## License

MIT
