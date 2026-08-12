import torch

def probs_from_logits(logits, temperature=1.0):
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        max_index = torch.argmax(logits, dim=-1)
        probs[max_index] = 1.0
        return probs
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return probs

def sample_from_probs(probs):
    return torch.multinomial(probs, num_samples=1).item()