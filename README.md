# TreeSpark

Calibrated, load-adaptive draft trees for semi-autoregressive speculative
decoding.

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

## Layout

```
treespark/
  tree.py              the method: calibrated best-first expansion, theta stop,
                       without-replacement sampling + recompute verification
  dspark.py            the semi-autoregressive drafter (backbone + Markov head)
  tree_speculative.py  tree attention mask + verification walk
  sampling.py          temperature sampling helpers
  baseline.py          target-only decode loop (reference/baseline)
```

## Paper

*TreeSpark: Calibrated, Load-Adaptive Draft Trees for Semi-Autoregressive
Speculative Decoding.* Raw experiment artifacts and the reproduction manual
are attached to this repository's
[releases](https://github.com/PopSoda2002/TreeSpark/releases).

## License

MIT
