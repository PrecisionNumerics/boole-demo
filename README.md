# Boole 1.7B inference

Run the public 1-bit Qwen3-1.7B model with **llama.cpp** or **vLLM + Humming**.
The model is currently hosted at [pn-admin/abc](https://huggingface.co/pn-admin/abc).
That Hugging Face path is temporary and will be renamed later; update `MODEL_REPO` when it changes.

```bash
MODEL_REPO=pn-admin/abc
python -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
```

## llama.cpp

Download the ready-to-run GGUF:

```bash
hf download "$MODEL_REPO" --include '*-Q1_0.gguf' --local-dir model
```

Build the public llama.cpp version with native Q1_0 support (requires Git, CMake, a C++ compiler,
and a CUDA toolkit for GPU inference):

```bash
git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
git -C third_party/llama.cpp checkout b42b7e6d3059503dd4bed6ca5dcf342bb6379f67
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build third_party/llama.cpp/build --config Release -j \
    --target llama-server

third_party/llama.cpp/build/bin/llama-server \
    -m model/*-Q1_0.gguf -ngl 99 -c 4096 --alias boole --host 127.0.0.1 --port 8000 \
    --chat-template-kwargs '{"enable_thinking": false}'
```

For CPU inference, build with `-DGGML_CUDA=OFF` and use `-ngl 0`.

## vLLM + Humming

From this repository, in the activated environment:

```bash
python -m pip install -e '.[vllm]'
boole-prep vllm "$MODEL_REPO" --out out/humming
vllm serve out/humming --served-model-name boole \
    --dtype bfloat16 --max-model-len 4096 --host 127.0.0.1 --port 8000 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

`boole-prep` downloads the packed checkpoint and exports a local Humming model directory.
It uses the model repository's config and tokenizer when provided, otherwise the public
`Qwen/Qwen3-1.7B` config and tokenizer. Installing this package registers the vLLM plugin
that handles the model's 1-bit embedding and output head. The packed checkpoint requires this
export before serving. You can also pass a downloaded model directory instead of `MODEL_REPO`.

Uses the public `vllm==0.31.0` and `humming-kernels==0.1.16` packages. The GPU path has been
checked on an RTX 5090 with bf16 activations and one GPU.

## Send a request

Both commands above expose an OpenAI-compatible API at `http://127.0.0.1:8000`:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"boole","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":64,"temperature":0}'
```
