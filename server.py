# Generates completions from RWKV model based on a prompt.

assert __name__ == "__main__", "This is a top-level script"

import os, sys, argparse
from multiprocessing import Lock

from tqdm import tqdm
import numpy as np
from flask import Flask, request
from flask_cors import CORS
import umsgpack

from util import array_to_bytes, bytes_to_array

sys.path.insert(0, "rwkv.cpp")

from rwkv.cpp_model import RWKVModel
from rwkv.cpp_shared_library import load_rwkv_shared_library

# =================================================================================================

parser = argparse.ArgumentParser(
    description="Generate completions from RWKV model based on a prompt"
)
parser.add_argument("model_path", help="Path to RWKV model in ggml format")
parser.add_argument("--host", type=str, help="flask host")
parser.add_argument("--port", type=int, help="flask port")
args = parser.parse_args()

try:
    library = load_rwkv_shared_library()
except OSError as e:
    print(e)
    print()
    print("no rwkv shared library")
    print("Please run `make` inside directory `rwkv.cpp/` to create the shared library")
    exit(1)
print(f"System info: {library.rwkv_get_system_info_string()}")

print("Loading RWKV model")
model = RWKVModel(library, args.model_path)

# ======================================== Server settings ========================================

api = Flask(__name__)
CORS(api)

@api.get("/info")
def get_model_info():
    return umsgpack.dumps(
        {
            "model_path": os.path.abspath(args.model_path),
            "vocab_count": model._logits_buffer_element_count,
            # seem to be n_layer * n_embed * 4
            "state_count": model._state_buffer_element_count,
        }
    )


logits_cache = None
infer_lock = Lock() # lock for global variables


@api.post("/infer")
def infer():
    global logits_cache
    data = umsgpack.loads(request.data)
    print(data.keys())

    tokens = data["tokens"]
    assert type(tokens) is list
    assert len(tokens) > 0, "tokens must be non-empty"

    state = data.get("state")
    if state is not None:
        state = bytes_to_array(data["state"])

    with infer_lock:
        for token in tqdm(tokens):
            logits_cache, state = model.eval(token, state, state, logits_cache)
        logits = array_to_bytes(logits_cache)

    return umsgpack.dumps(
        {
            "state": array_to_bytes(state),
            "logits": logits,
        }
    )


api.run(host=args.host, port=args.port)
