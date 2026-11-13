<p align="center">
  <img src="assets/logo-white-on-black.png" width="128" height="128" alt="Boole logo">
</p>

<h1 align="center">Boole Demo</h1>

<p align="center">
  <strong>Run Boole models on your own machine.</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/precision-numerics">🤗 Models</a>
  &nbsp;·&nbsp;
  <a href="https://precisionnumerics.ai/chat/">💬 On-Device Chat</a>
  &nbsp;·&nbsp;
  <a href="#quick-start">Quick start</a>
</p>

| Model | Runtime | Model file | Hardware |
| --- | --- | --- | --- |
| [Boole-1.7B](https://huggingface.co/precision-numerics/Boole-1.7B-GGUF) | [llama.cpp](#boole-17b-llamacpp) | [Boole-1.7B-Q1_0.gguf](https://huggingface.co/precision-numerics/Boole-1.7B-GGUF/blob/main/Boole-1.7B-Q1_0.gguf) (292.1 MB) | CPU or CUDA GPU |
| [Boole-1.7B](https://huggingface.co/precision-numerics/Boole-1.7B) | [vLLM + Humming](#boole-17b-vllm--humming) | [Boole-1.7B.safetensors](https://huggingface.co/precision-numerics/Boole-1.7B/blob/main/Boole-1.7B.safetensors) (247.1 MB), exported locally | CUDA GPU, bf16, single GPU |
| [Boole-32B-A2.5B-preview](https://huggingface.co/precision-numerics/Boole-32B-A2.5B-preview-GGUF) | [llama.cpp, our branch](#boole-32b-a25b-preview-llamacpp-our-branch) | [Boole-32B-A2.5B-preview-Q2_0.gguf](https://huggingface.co/precision-numerics/Boole-32B-A2.5B-preview-GGUF/blob/main/Boole-32B-A2.5B-preview-Q2_0.gguf) (8.3 GiB) | CUDA GPU |

Each model card has the evaluation and speed results. Boole-32B-A2.5B-preview needs our llama.cpp
branch, which this repository pins; stock llama.cpp cannot load it yet.

## Quick start

Python 3.10 or newer is required. Run the commands from a clone of this repository:

```bash
git clone https://github.com/PrecisionNumerics/boole-demo.git
cd boole-demo

python -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
```

### Boole-1.7B: llama.cpp

Download the GGUF, then fetch and build the pinned upstream llama.cpp:

```bash
hf download precision-numerics/Boole-1.7B-GGUF Boole-1.7B-Q1_0.gguf --local-dir model
bash scripts/setup_llama_cpp.sh            # CUDA build; GGML_CUDA=0 for CPU only
```

Start the server and open the URL it prints:

```bash
third_party/llama.cpp/build/bin/llama-server \
    -m model/Boole-1.7B-Q1_0.gguf \
    -ngl 99 -c 4096 \
    --reasoning off
```

Use `-ngl 0` on CPU. Boole-1.7B is meant to run with thinking off. Try: **Explain why the sky is blue in one sentence.**

### Boole-1.7B: vLLM + Humming

Install the loader, export the model, and start the server:

```bash
python -m pip install -e '.[vllm]'
boole-prep vllm precision-numerics/Boole-1.7B --out out/humming

vllm serve out/humming \
    --dtype bfloat16 --max-model-len 4096 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

In another terminal, send a chat request (replace `<server-url>` with the server's URL):

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

The reply is in `choices[0].message.content`.

`boole-prep` downloads the packed checkpoint and writes a vLLM model directory. Installing
the package registers a vLLM plugin that loads the 1-bit embedding and output head. The
safetensors repository includes the Qwen3-1.7B config and tokenizer files, so a downloaded
model directory can also be exported offline by passing its path instead of the repo ID.

The repo ID defaults to `precision-numerics/Boole-1.7B`; `boole-prep vllm --out out/humming`
exports the same model. The GGUF repository is for llama.cpp.

The loader targets **vLLM 0.31.0** and **humming-kernels 0.1.16**, both pinned by the `vllm` extra.

### Boole-32B-A2.5B-preview: llama.cpp (our branch)

Download the GGUF, then fetch and build our pinned llama.cpp branch (CUDA only; it builds into
`third_party/llama.cpp-boole-32b`, next to the Boole-1.7B build):

```bash
hf download precision-numerics/Boole-32B-A2.5B-preview-GGUF Boole-32B-A2.5B-preview-Q2_0.gguf --local-dir model
MODEL=boole-32b bash scripts/setup_llama_cpp.sh
```

Start the server and open the URL it prints:

```bash
GGML_CUDA_GRAPH_OPT=1 third_party/llama.cpp-boole-32b/build/bin/llama-server \
    -m model/Boole-32B-A2.5B-preview-Q2_0.gguf \
    -ngl 99 -fa on -ub 1024 -c 32768
```

It needs about 10.3 GB of GPU memory at this 32K context. The model reasons before it answers, and the
reasoning is returned separately from the reply. Try: **Write a Python function that returns the n-th
Fibonacci number iteratively.**

## Repository layout

| Path | Contents |
| --- | --- |
| `src/boole_demo/` | Checkpoint reader, vLLM export, vLLM plugin and the `boole-prep` command |
| `scripts/` | `setup_llama_cpp.sh`: fetch and build the pinned llama.cpp (upstream for Boole-1.7B, our branch for Boole-32B-A2.5B-preview with `MODEL=boole-32b`) |
| `tests/` | CPU-only tests on a synthetic checkpoint (`pip install -e '.[dev]'`, then `pytest`) |
| `assets/` | Logo |

## License

Runtime code and model weights are licensed under **Apache License 2.0**. See the
[runtime license](LICENSE) and the
model licenses for [Boole-1.7B](https://huggingface.co/precision-numerics/Boole-1.7B/blob/main/LICENSE) and
[Boole-32B-A2.5B-preview](https://huggingface.co/precision-numerics/Boole-32B-A2.5B-preview-GGUF/blob/main/LICENSE).
