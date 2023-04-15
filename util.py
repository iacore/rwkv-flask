import numpy as np

def tensor_to_bytes(tensor):
    import torch
    assert type(tensor) is torch.Tensor
    return bytes(np.array(tensor))

def bytes_to_tensor(b):
    import torch
    assert type(b) is bytes
    return torch.frombuffer(b, dtype=torch.float32)

def sample_logits(logits, temperature=1.0, top_p=0.85):
    e_x = np.exp(logits - np.max(logits))
    probs = e_x / e_x.sum()  # Softmax of x

    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = sorted_probs[np.argmax(cumulative_probs > top_p)]
    probs[probs < cutoff] = 0
    probs = probs ** (1 / temperature)
    return np.random.choice(a=len(probs), p=probs / np.sum(probs))
