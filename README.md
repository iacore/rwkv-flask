# rwkv-flask

Stateless HTTP server for rwkv.cpp. Tokenizer not included. Request cache not included.

```shell
## build librwkv.so
git submodule init
git submodule update --depth 1 --recursive
pushd rwkv.cpp
git submodule init
git submodule update --depth 1 --recursive
rm -r build/ ; cmake -B build BUILD_SHARED_LIBS=ON && cmake --build build
cp build/librwkv.so ./
popd


## Python

# install deps
pdm install -d

# run server
pdm run python server.py ../models/RWKV-4-Raven-3B-v9-Eng99\%-Other1\%-20230411-ctx4096-q41.bin

# run client
pdm run python example_client.py
```

If you don't want to use pdm, install the dependencies in [pyproject.toml](pyproject.toml) with `pip`.

> **Note**
> please use a msgpack impl. that has support for raw bytes.
>
> For Python: https://pypi.org/project/u-msgpack-python/  
> For Javascript (Deno/Node): https://esm.sh/@msgpack/msgpack@2.8.0

