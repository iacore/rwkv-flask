# rwkv-flask

Stateless HTTP server for rwkv.cpp. Tokenizer not included.

```shell
# install deps
pdm install -d

# run server
pdm run python server.py ../models/RWKV-4-Raven-3B-v9-Eng99\%-Other1\%-20230411-ctx4096-q41.bin

# run client
pdm run python example_client.py
```

Tip: please use a msgpack impl. that has support for raw bytes.

For Python: https://pypi.org/project/u-msgpack-python/  
For Javascript (Deno/Node): https://esm.sh/@msgpack/msgpack@2.8.0

## TODO

- cache requests
