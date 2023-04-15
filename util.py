import numpy as np


def array_to_bytes(tensor):
    assert type(tensor) is np.ndarray
    return bytes(tensor)


def bytes_to_array(b):
    assert type(b) is bytes
    return np.frombuffer(b, dtype=np.float32)


def sample_logits(logits, temperature=1.0, top_p=0.85):
    e_x = np.exp(logits - np.max(logits))
    probs = e_x / e_x.sum()  # Softmax of x

    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = sorted_probs[np.argmax(cumulative_probs > top_p)]
    probs[probs < cutoff] = 0
    probs = probs ** (1 / temperature)
    return np.random.choice(a=len(probs), p=probs / np.sum(probs))
