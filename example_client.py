import os
import pathlib
import time

import numpy as np
import tokenizers
import requests
import umsgpack

from util import sample_logits


def infer_remote(tokens, state):
    resp = requests.post(
        "http://127.0.0.1:5000/infer",
        data=umsgpack.dumps(
            {
                "tokens": tokens,
                "state": state,
            }
        ),
    )
    assert resp.ok, resp
    data = umsgpack.loads(resp.content)
    assert type(data["logits"]) is bytes
    assert type(data["state"]) is bytes
    return np.frombuffer(data["logits"], dtype=np.float32), data["state"]


print("Loading 20B tokenizer")
tokenizer_path = pathlib.Path(os.path.abspath(__file__)).parent / "20B_tokenizer.json"
tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))

# ======================================== Script settings ========================================

prompt: str = """# rwkv.cpp

This is a port of [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) to [ggerganov/ggml](https://github.com/ggerganov/ggml).

Besides usual **FP32**, it supports **FP16** and **quantized INT4** inference on CPU. This project is **CPU only**."""

# Token count per single completion.
tokens_per_generation: int = 100

# Sampling settings.
temperature: float = 0.8
top_p: float = 0.5

# =========================================== Run Model ===========================================

prompt_tokens = tokenizer.encode(prompt).ids
prompt_token_count = len(prompt_tokens)
assert prompt_token_count != 0, "Prompt must not be empty"
print(f"{prompt_token_count} tokens in prompt")

state = None
logits, state = infer_remote(prompt_tokens, state)

# generation start
print(f"\n--- Generation n_token={tokens_per_generation} ---\n")
print(prompt, end="[")
start = time.time()

for i in range(tokens_per_generation):
    token = sample_logits(logits, temperature, top_p)

    print(tokenizer.decode([token]), end="", flush=True)

    logits, state = infer_remote([token], state)

delay = time.time() - start
print(
    "]\n\nTook %.3f sec, %d ms per token"
    % (delay, delay / tokens_per_generation * 1000)
)
