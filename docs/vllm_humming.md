# vLLM + Humming

## Exported tensors (`boole-prep vllm`)

Every 1-bit module uses Humming's native `uint1` schema with an fp zero point, which turns
`±m` into `(q − 0.5) · 2m`:

```text
<name>.weight        int32 [N, K/32]   the source sign bytes reinterpreted (little-endian, LSB-first)
<name>.weight_scale  bf16  [N, K/G]    2 · max(|scale|, 1e-5)
```

No bit is re-ordered, and with a zero point of 0.5 Humming's own `dequantize_weight` reads these
tensors back as exactly the source weights (tested). The zero point is the same constant for every
group, so it is **not stored** — the plugin supplies it, and the exported tensors total exactly the
source file's bytes (tested). `model.embed_tokens` keeps the only code tensor; `lm_head` ships just
its own `weight_scale`. `config.json` is Qwen3's with `tie_word_embeddings: true` and

```json
"quantization_config": {"quant_method": "w1_humming", "dtype": "uint1", "group_size": 128,
                        "has_zero_point": true, "is_fp_zero_point": true, "zero_point": 0.5,
                        "scale_floor": 1e-5}
```

## Why a plugin

vLLM 0.31's Humming backend serves `Linear` layers only: `humming_is_layer_skipped` always skips
`lm_head`, and the config has no method for `VocabParallelEmbedding`. `W1HummingConfig`
(`src/boole_demo/vllm_quant.py`) subclasses `HummingConfig`, delegates every linear layer to the
stock Humming linear method (so TP sharding, QKV/gate-up fusion, tuning, compile and CUDA graphs are
upstream's) through `W1LinearMethod`, a subclass whose only change is filling the zero point with
the constant 0.5 when the parameter is created, and adds:

- `W1EmbeddingMethod.embedding` — gathers rows of the int32 code tensor, views them as bytes,
  unpacks the bits and applies `(bit − 0.5) · scale` with the *embedding's* group scales.
- `W1HeadMethod` — creates the Humming parameters on `ParallelLMHead`; its `tie_weights`
  (vLLM calls it through `ParallelLMHead.tie_weights` for tied models) makes `lm_head.weight` the
  embedding's code Parameter while `weight_scale` stays the head's own and loads from
  `lm_head.weight_scale`. After loading, the stock method repacks the head into Humming's kernel layout;
  the embedding keeps its raw codes. Humming pads N to a multiple of 256 (151936 → 152064), so
  `apply` slices the logits back to the layer's width.

The package registers `w1_humming` via the `vllm.general_plugins` entry point (`boole_w1`).

## Verified / not verified

Verified on one RTX 5090: offline `LLM` and `vllm serve`, compiled + CUDA graphs and eager, module
wiring (`W1EmbeddingMethod` / `W1HeadMethod` / `W1LinearMethod`, head scales ≠ embedding scales),
logits against a dense reference (README table; tiny-model max |Δlogprob| 0.023 compiled, 0.026
eager, tolerance 0.06).

Not verified: tensor parallel > 1 (the vocab/linear loaders are upstream's, the plugin adds no TP
code), fp16 activations, GPUs other than SM120, other vLLM versions (the plugin imports
`vllm.model_executor.layers.quantization.humming`, which is not a public API — hence `vllm==0.31.0`).
