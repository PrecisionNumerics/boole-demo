# boole-demo

Prepare the `opd1200` 1-bit Qwen3-1.7B checkpoint
(`qwen3-1p7b-SFTv3-1000M-opd1200-packed.safetensors`) for inference with

1. **llama.cpp** — a GGUF whose every 1-bit tensor is llama.cpp's native `Q1_0`, and
2. **vLLM + Humming** — a vLLM model directory served by vLLM's Humming kernels, with a small
   plugin that adds the 1-bit embedding and output head stock vLLM cannot serve.

The repository is self-contained: it depends only on public packages (`torch`, `safetensors`,
`gguf`, `vllm`, `humming-kernels`) and a pinned upstream llama.cpp checkout.

## The checkpoint in one paragraph

Every projection, the input embedding and the output head store **1 sign bit per weight**
(`uint8 [N, K/8]`, LSB-first) plus a **bf16 scale per group of 128 inputs** (`[N, K/128]`); the
effective weight is `±max(|scale|, 1e-5)`. The embedding and the head **share one sign tensor but
have different scales** — they are two modules over identical bits. Details and evidence:
[docs/format.md](docs/format.md).

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[vllm,test]'          # vllm==0.31.0 pulls torch 2.13 + humming-kernels
scripts/setup_llama_cpp.sh                # pinned llama.cpp, CUDA build -> third_party/llama.cpp

CKPT=/workspace/qwen3-1p7b-SFTv3-1000M-opd1200-packed.safetensors
boole-prep inspect $CKPT                  # validate + inventory
boole-prep all $CKPT --out-dir out        # writes both artifacts below (~20 s)
```

`boole-prep all` produces

| artifact | size | used by |
|---|---|---|
| `out/<stem>-Q1_0.gguf` | 292 MB | llama.cpp |
| `out/<stem>-humming/` (`model.safetensors` 247 MB + config, tokenizer) | 262 MB | vLLM |

The checkpoint has no config or tokenizer, so they come from the public `Qwen/Qwen3-1.7B` repo
(shapes match it exactly). Pass `--tokenizer DIR_OR_REPO` / `--base-config FILE` to override.

### Why the GGUF is bigger than the source

292.1 MB = 247.1 MB source + 38.9 MB (`output.weight` needs its own copy of the embedding's sign
bits, since it has different scales and GGUF tensors cannot share bytes) + 5.95 MB tokenizer
vocabulary/merges (the safetensors has no tokenizer) + 0.25 MB (norm weights stored as F32, as
llama.cpp expects). The vLLM `model.safetensors` is exactly the source size: it stores the shared
code tensor once and nothing else beyond the (rescaled) scales.

### llama.cpp

```bash
B=third_party/llama.cpp/build/bin
$B/llama-completion -m out/*-Q1_0.gguf -ngl 99 -no-cnv --temp 0 -n 48 -p "The capital of France is"
$B/llama-server -m out/*-Q1_0.gguf -ngl 99 -c 4096 \
    --chat-template-kwargs '{"enable_thinking": false}'          # OpenAI-style /v1 API
```

### vLLM + Humming

```bash
vllm serve out/<stem>-humming --served-model-name opd1200 --max-model-len 4096
```

The plugin registers itself through the `vllm.general_plugins` entry point when the package is
installed in the same environment — no flags needed. Use bf16 (the default for this config).
See [docs/vllm_humming.md](docs/vllm_humming.md).

## How the tied embedding is handled

| | embedding | output head |
|---|---|---|
| source | `model.embed_tokens.weight` + `.weight_scale` | same sign tensor (metadata alias) + **its own** `lm_head.weight_scale` |
| llama.cpp | `token_embd.weight` (Q1_0) | `output.weight` (Q1_0): identical sign bytes, own scale blocks (+39 MB: llama.cpp's loader rejects tensors sharing an offset) |
| vLLM | `W1EmbeddingMethod`: gather packed rows, apply own scale | `W1HeadMethod`: `tie_weights` shares the *codes*, keeps own scale, then the stock Humming GEMM |

## Verification

Run on one RTX 5090 (SM120), driver 580.105.08, CUDA 13.0, torch 2.13.0+cu130, vLLM 0.31.0,
humming-kernels 0.1.16, llama.cpp `b42b7e6d`, gguf 0.19.0.

**Weights** (`verify_gguf`, tests): all 198 quantized GGUF tensors dequantize to exactly
`sign × fp16(max(|scale|, 1e-5))` and all 113 dense tensors equal the source; Humming's own
reference dequantiser reproduces the source weights bit-exactly from the exported tensors. The only
rounding anywhere is bf16→fp16 on the 58,899 of 15,872,000 scales that sit below the floor
(≤ 0.14 % relative).

**Logits** — `scripts/compare_engines.py`, 26 prompts (plain, chat-templated, and prefixes of one
passage), identical token ids fed to all three engines, dense bf16 transformers model as reference:

| engine | top-1 = reference | mean abs Δlogprob (ref top-5) | max abs Δ | greedy prefix match (of 32) | identical 32 tokens | passage NLL |
|---|---|---|---|---|---|---|
| dense bf16 reference | – | – | – | – | – | 2.2966 |
| vLLM + Humming | 26/26 | 0.0344 | 0.2445 | 23.5 | 12/26 | 2.2947 |
| llama.cpp Q1_0 | 26/26 | 0.0323 | 0.1486 | 25.7 | 18/26 | 2.3151 |

Also: llama.cpp tokenizes 15/15 probe strings (special tokens, digits, whitespace, CJK, emoji,
code) identically to HF; `vllm serve` answers `/v1/chat/completions`; vLLM runs with
`torch.compile` and CUDA graphs (the tiny-model engine tests cover compiled and eager).

vLLM is not run-to-run deterministic here (continuous batching changes kernel accumulation order):
re-running the *same* export gave 492/518 compared log-probs that differ (max |Δ| 0.18) and 5/26
different greedy sequences, so the vLLM max-Δ and greedy columns move by that much between runs.

Read these numbers with care: 26 prompts is a smoke-level sample, and the NLL column covers only the
9 passage positions where every engine reported the true token (llama.cpp returns top-20 only).
Greedy continuations diverge late in about a third of prompts for both engines; llama.cpp quantizes
activations to 8 bits for `Q1_0` mat-muls while Humming and the reference keep 16-bit activations,
which is consistent with — but was not isolated as — the source of the small differences.
No perplexity benchmark or downstream eval has been run on either engine.

## Tests

```bash
pytest -q                                   # unit + tiny-model vLLM engine tests (GPU)
BOOLE_CHECKPOINT=$CKPT BOOLE_TOKENIZER=Qwen/Qwen3-1.7B LLAMA_CPP_DIR=third_party/llama.cpp pytest -q
python scripts/compare_engines.py all --checkpoint $CKPT --vllm-dir out/<stem>-humming \
    --gguf out/<stem>-Q1_0.gguf --out-dir out/compare
```

The engine tests build a tiny synthetic checkpoint with the real model's traps (tied codes with
distinct scales, negative and below-floor scales, a vocabulary that is not a multiple of vLLM's
or Humming's tile) and include two negative controls — signed scales and a head reusing the
embedding's scales — that must miss the dense reference by > 20× the pass tolerance.

## Known limits

- **Chat template / tokenizer** come from `Qwen/Qwen3-1.7B`; the checkpoint does not say whether SFT
  changed them. If it did, pass `--tokenizer`.
- **vLLM is pinned to 0.31.0**: the plugin subclasses `vllm.model_executor.layers.quantization.humming`
  (not a stable API). Tensor parallelism > 1, fp16 activations, and non-SM120 GPUs are untested.
- **Only bf16 activations** are validated for vLLM. fp16 would round the sub-floor scales into fp16
  subnormals.
- The head's Humming copy and the embedding's raw codes coexist on the GPU (~39 MB extra).
- Throughput has not been measured; no performance claims are made here.
