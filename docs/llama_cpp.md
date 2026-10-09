# llama.cpp (Q1_0 GGUF)

llama.cpp's `Q1_0` block (type 41, `QK1_0 = 128`) is this checkpoint's own layout:

```text
block_q1_0 { fp16 d; uint8 qs[16]; }      // 18 bytes per 128 weights
weight j = (qs[j/8] >> (j%8)) & 1 ? +d : -d
```

so the export is a re-wrap, not a re-quantization: the 16 sign bytes are copied unchanged and
`d = fp16(max(|scale|, 1e-5))`. `token_embd.weight` and `output.weight` are written as two Q1_0
tensors with identical sign bytes and their own `d`. Norm weights are F32 (llama.cpp's convention).

Metadata and tokenizer are produced by llama.cpp's own `convert_hf_to_gguf.py --vocab-only` on a
stub directory (config + tokenizer files) and copied key-for-key, so they are whatever the pinned
llama.cpp writes for a dense Qwen3 (`tokenizer.ggml.pre = qwen2`, chat template included);
`general.file_type` is set to `MOSTLY_Q1_0` (40).

`boole-prep gguf` re-reads the file and checks each tensor against the source: Q1_0 tensors must
dequantize to exactly `sign × fp16(magnitude)`, dense tensors must be equal, and no tensor may be
missing or extra.

## Pinned build

`scripts/setup_llama_cpp.sh` fetches commit `b42b7e6d3059503dd4bed6ca5dcf342bb6379f67` and builds
`llama-cli`, `llama-completion`, `llama-server`, `llama-bench`, `llama-perplexity` with CUDA for the
local GPU's architecture (`GGML_CUDA=0` for CPU only). `Q1_0` has CPU, `get_rows` (embedding) and
CUDA mat-mul support at that commit. Loading prints "control-looking token 128247 '</s>' was not
control-type"; it concerns the tokenizer metadata exactly as the stock converter wrote it, was not
investigated, and did not affect any check in the README.

llama.cpp quantizes activations to 8 bits for `Q1_0` mat-muls, so its numerics are W1A8, unlike the
W1A16 of the vLLM path.
