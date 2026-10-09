<h1 align="center">Boole Demo</h1>

<p align="center">
  <strong>1-bit Qwen3-1.7B, running locally.</strong><br>
  Inference with llama.cpp or vLLM + Humming using public runtimes.
</p>

<p align="center">
  <a href="https://huggingface.co/pn-admin/abc"><strong>🤗 Hugging Face model</strong></a>
  &nbsp;·&nbsp;
  <a href="#quick-start">Quick start</a>
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

Choose a runtime below. The examples use a 4,096-token context and disable thinking.

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

Requires Git, CMake, a C++ compiler, and a CUDA toolkit.

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
    -ngl 99 -c 4096 \
    --chat-template-kwargs '{"enable_thinking": false}'
```

### vLLM + Humming

Install the loader, prepare the model, and start the server:

```bash
python -m pip install -e '.[vllm]'
boole-prep vllm "$MODEL_REPO" --out out/humming

vllm serve out/humming \
    --dtype bfloat16 --max-model-len 4096 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

`boole-prep` downloads and exports the packed checkpoint. Installing the package registers the
loader for its 1-bit embedding and output head. Config and tokenizer files come from the model
repository when available, otherwise the public `Qwen/Qwen3-1.7B` assets. A downloaded model
directory can also be passed in place of `MODEL_REPO`.

The loader targets **vLLM 0.31.0** and **humming-kernels 0.1.16**, pinned by the package.

## License

Licensed under the [Apache License 2.0](LICENSE).
