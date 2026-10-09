<h1 align="center">Boole Demo</h1>

<p align="center">
  <strong>1-bit Qwen3-1.7B, running locally.</strong><br>
  Inference with llama.cpp or vLLM + Humming using public runtimes.
</p>

<p align="center">
  <a href="https://huggingface.co/pn-admin/abc"><strong>🤗 Hugging Face model</strong></a>
  &nbsp;·&nbsp;
  <a href="#quick-start">Quick start</a>
  &nbsp;·&nbsp;
  <a href="#api-example">API example</a>
</p>

---

| Runtime | Model format | Hardware |
|---|---|---|
| [llama.cpp](#llamacpp) | Published Q1_0 GGUF | CPU or CUDA GPU |
| [vLLM + Humming](#vllm--humming) | Packed checkpoint → local Humming export | CUDA GPU; bf16, single GPU |

The model is currently hosted at **`pn-admin/abc`**. This path is temporary and will be renamed
later. Change `MODEL_REPO` below when the new name is available.

## Quick start

Run these commands from a clone of this repository. Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub

export MODEL_REPO=pn-admin/abc
```

Choose one of the following runtimes. Both serve the model as `boole` at
**http://127.0.0.1:8000** with a 4,096-token context and thinking disabled.

### llama.cpp

Download the GGUF, which is ready to load without conversion:

```bash
hf download "$MODEL_REPO" --include '*-Q1_0.gguf' --local-dir model
```

Check out the pinned upstream runtime, then choose a CUDA or CPU build below:

```bash
git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
git -C third_party/llama.cpp checkout b42b7e6d3059503dd4bed6ca5dcf342bb6379f67
```

<details>
<summary><strong>Build llama.cpp with CUDA</strong></summary>

Requires Git, CMake, a C++ compiler, and a CUDA toolkit. This pins the upstream version
used to check the model's native Q1_0 support.

```bash
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build third_party/llama.cpp/build --config Release -j \
    --target llama-server
```

</details>

<details>
<summary><strong>CPU inference</strong></summary>

Configure and build without CUDA:

```bash
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build third_party/llama.cpp/build --config Release -j \
    --target llama-server
```

Use `-ngl 0` in the server command below.

</details>

Start the server:

```bash
third_party/llama.cpp/build/bin/llama-server \
    -m model/*-Q1_0.gguf \
    -ngl 99 -c 4096 --alias boole \
    --host 127.0.0.1 --port 8000 \
    --chat-template-kwargs '{"enable_thinking": false}'
```

Open **http://127.0.0.1:8000** to use llama.cpp's built-in chat interface, or send an
[API request](#api-example).

### vLLM + Humming

Install the loader, prepare the model, and start the server:

```bash
python -m pip install -e '.[vllm]'
boole-prep vllm "$MODEL_REPO" --out out/humming

vllm serve out/humming \
    --served-model-name boole \
    --dtype bfloat16 --max-model-len 4096 \
    --host 127.0.0.1 --port 8000 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

`boole-prep` downloads the packed checkpoint and exports it for Humming. Installing the
package also registers the vLLM loader for the model's 1-bit embedding and output head.
The packed checkpoint must be exported before it can be served.

<details>
<summary><strong>Versions, tokenizer, and local model files</strong></summary>

- The loader uses the public **vLLM 0.31.0** and **humming-kernels 0.1.16** packages.
- Config and tokenizer files come from the model repository when present. For a weights-only
  repository, they come from the public `Qwen/Qwen3-1.7B` model.
- The CUDA path has been checked on an RTX 5090 with bf16 activations and a single GPU.
- To prepare a model already downloaded to disk:

  ```bash
  boole-prep vllm /path/to/model --out out/humming
  ```

</details>

## API example

With either server running, send a chat request from another terminal:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "boole",
      "messages": [{"role": "user", "content": "What is 2 + 2?"}],
      "max_tokens": 64,
      "temperature": 0
    }'
```

Both runtimes expose an OpenAI-compatible API. The example was checked with both engines;
the assistant returned `2 + 2 = 4.`
