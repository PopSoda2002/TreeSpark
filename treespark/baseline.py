import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from treespark.sampling import probs_from_logits, sample_from_probs

# Generic HF pair used only by the module self-tests here and in
# tree_speculative.py; the paper's drafter is DSpark (see README quickstart).
TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_DEVICE = "cuda"

def load_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    return model

def build_prompt(tokenizer, user_message, device):
    text = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": user_message
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt").input_ids.to(device)

@torch.no_grad()
def baseline_generate(model, input_ids, max_new_tokens, temperature, eos_id=None):
    ids = input_ids.clone()
    target_forward = 0
    for _ in range(max_new_tokens):
        logits = model(ids).logits
        last = logits[0, -1]
        probs = probs_from_logits(last, temperature)
        next_id = sample_from_probs(probs)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device, dtype=ids.dtype)], dim=-1)
        target_forward += 1
        if eos_id is not None and next_id == eos_id:
            break
    return ids, {"target_forward": target_forward}

@torch.no_grad()
def baseline_generate_cached(model, input_ids, max_new_tokens, temperature, eos_id=None):
    ids = input_ids.clone()
    target_forward = 0
    cache = DynamicCache()
    # Prefill
    logits = model(ids, past_key_values=cache).logits
    for _ in range(max_new_tokens):
        last = logits[0, -1]
        probs = probs_from_logits(last, temperature)
        next_id = sample_from_probs(probs)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device, dtype=ids.dtype)], dim=-1)
        target_forward += 1
        if eos_id is not None and next_id == eos_id:
            break
        # Decode
        logits = model(torch.tensor([[next_id]], device=ids.device, dtype=ids.dtype), past_key_values=cache).logits
    return ids, {"target_forward": target_forward}

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    model = load_model(TARGET_MODEL, TARGET_DEVICE)
    input_ids = build_prompt(tokenizer, "Hello, how are you?", TARGET_DEVICE)
    ids, metrics = baseline_generate(model, input_ids, max_new_tokens=100, temperature=0.0, eos_id=tokenizer.eos_token_id)
    print(tokenizer.decode(ids[0, input_ids.shape[1]:], skip_special_tokens=True))
    print(metrics)
