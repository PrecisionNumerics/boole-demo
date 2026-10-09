<h1 align="center">Boole Demo</h1>

<p align="center">
  <strong>1-bit Qwen3-1.7B, running locally.</strong><br>
  Inference with llama.cpp or vLLM + Humming using public runtimes.
</p>

<p align="center">
  <a href="https://huggingface.co/precision-numerics/Boole-1.7B"><strong>🤗 Hugging Face model</strong></a>
  &nbsp;·&nbsp;
  <a href="#quick-start">Quick start</a>
</p>

---

| Runtime | Model format | Hardware |
|---|---|---|
| [llama.cpp](#llamacpp) | Published Q1_0 GGUF | CPU or CUDA GPU |
| [vLLM + Humming](#vllm--humming) | Packed checkpoint → local Humming export | CUDA GPU; bf16, single GPU |

The model is hosted at **`precision-numerics/Boole-1.7B`**.
The model repository is currently private; use a Hugging Face account with access.

## Quick start

Run these commands from a clone of this repository. Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
hf auth login

export MODEL_REPO=precision-numerics/Boole-1.7B
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

Once the server is ready, open the URL printed in the terminal in your browser.
In the chat UI, try: **Explain why the sky is blue in one sentence.**

### vLLM + Humming

Install the loader, prepare the model, and start the server:

```bash
python -m pip install -e '.[vllm]'
boole-prep vllm "$MODEL_REPO" --out out/humming

vllm serve out/humming \
    --dtype bfloat16 --max-model-len 4096 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

Keep the server running. In another terminal, replace `<server-url>` with the server's URL
and send a chat request:

```bash
curl -sS '<server-url>/v1/chat/completions' \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "out/humming",
      "messages": [{"role": "user", "content": "Explain why the sky is blue in one sentence."}],
      "max_tokens": 128,
      "temperature": 0
    }'
```

The reply is in `choices[0].message.content`. Change the message to ask your own question.

`boole-prep` downloads and exports the packed checkpoint. Installing the package registers the
loader for its 1-bit embedding and output head. This release includes the Qwen3-1.7B config
and tokenizer files, so a downloaded model directory can also be exported offline in place
of `MODEL_REPO`.

The loader targets **vLLM 0.31.0** and **humming-kernels 0.1.16**, pinned by the package.

## License

Runtime code is licensed under [Apache License 2.0](LICENSE).
Model weights in this release use the
[PolyForm Small Business License 1.0.0](https://huggingface.co/precision-numerics/Boole-1.7B/blob/main/LICENSE),
with an additional grant for personal, noncommercial use by individuals.
