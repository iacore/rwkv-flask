import numpy as np


def hash_file(path: str) -> bytes:
    from xxhash import xxh3_128
    with open(path, "rb") as f_model:
        model_hash_obj = xxh3_128()
        while True:
            some_data = f_model.read(4096)
            if len(some_data) == 0:
                break
            model_hash_obj.update(some_data)
        return model_hash_obj.digest()


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

from time import time
from collections import OrderedDict

class TimeBoundedLRU:
    "LRU Cache that invalidates and refreshes old entries."

    def __init__(self, func, maxsize=128, maxage=30):
        self.cache = OrderedDict()      # { args : (timestamp, result)}
        self.func = func
        self.maxsize = maxsize
        self.maxage = maxage

    def __call__(self, *args):
        if args in self.cache:
            self.cache.move_to_end(args)
            timestamp, result = self.cache[args]
            if time() - timestamp <= self.maxage:
                return result
        result = self.func(*args)
        self.cache[args] = time(), result
        if len(self.cache) > self.maxsize:
            self.cache.popitem(0)
        return result
