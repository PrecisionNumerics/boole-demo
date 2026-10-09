# Checkpoint format

`qwen3-1p7b-SFTv3-1000M-opd1200-packed.safetensors` (247 MB, 508 tensors, no config/tokenizer).

## Layout

For each quantized module `<name>` (196 projections, `model.embed_tokens`, `lm_head`):

```text
<name>.weight        uint8 [N, K/8]     bit k%8 of byte k//8 is weight k;  1 -> +1,  0 -> -1
<name>.weight_scale  bf16  [N, K/128]   one scale per 128 input weights
```

Down-projections have `K = 6144` (48 groups); everything else `K = 2048` (16 groups). The 113
remaining tensors are bf16 RMSNorm weights (`input_layernorm`, `post_attention_layernorm`,
`q_norm`, `k_norm`, `model.norm`).

`lm_head.weight` is absent; the safetensors `__metadata__` aliases it to
`model.embed_tokens.weight` (`{"lm_head.weight": "model.embed_tokens.weight"}`). The reader
requires that alias to be explicit and allows it for the head only.

## Effective weight

```text
W[n, k] = sign(bit[n, k]) * max(|weight_scale[n, k // 128]|, 1e-5)
```

The stored scales are *raw learned* values, not magnitudes: 676,380 of 15,872,000 are negative,
58,899 have |s| < 1e-5, and the largest is 41.25 (an embedding scale). The sign of a stored scale
carries no information; the exports bake `abs` and the floor in, because neither llama.cpp nor
Humming applies them.

**Evidence** that `abs` is the right reading (dense bf16 model, 127-token news passage,
`src/boole_demo/reference.py`): NLL/token **2.66** (ppl 14.3) with `max(|s|, 1e-5)` versus
**16.34** (ppl 1.2e7) if negative scales flip the sign. The floor itself moves 0.4 % of scales by
at most 1e-5 and was not separately ablated.

## Shared embedding, distinct scales

The embedding and head have identical sign bytes and different scales (max |Δ| = 41.25;
4,447 and 7,264 distinct values — not a small codebook). They are therefore two modules over one
code tensor. Every export keeps the scales separate; see the table in the README.

## Dimensions

Inferred from shapes and checked against `Qwen/Qwen3-1.7B`: hidden 2048, 28 layers, 16 heads /
8 KV heads, head_dim 128, intermediate 6144, vocab 151936. `infer_dims` also requires the exact
tensor inventory of a dense Qwen3 (any extra or missing tensor is an error).
