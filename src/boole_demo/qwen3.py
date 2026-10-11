"""Check the packed checkpoint against the model repository's Qwen3 configuration."""

from __future__ import annotations

from .w1 import W1Checkpoint, W1FormatError

_LAYER_LINEARS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_LAYER_DENSE = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)


def infer_dims(ckpt: W1Checkpoint) -> dict:
    """Dimensions implied by tensor shapes, after checking the tensor inventory is exactly
    that of a dense Qwen3 whose projections, embedding and head are 1-bit."""
    by_name = {m.name: m for m in ckpt.modules}
    layers = sorted(
        {int(m.name.split(".")[2]) for m in ckpt.modules if m.name.startswith("model.layers.")}
    )
    if not layers or layers != list(range(len(layers))):
        raise W1FormatError(f"decoder layers are not contiguous from 0: {layers[:5]}...")
    expected_modules = {"model.embed_tokens", "lm_head"}
    expected_dense = {"model.norm.weight"}
    for i in layers:
        expected_modules |= {f"model.layers.{i}.{p}" for p in _LAYER_LINEARS}
        expected_dense |= {f"model.layers.{i}.{p}" for p in _LAYER_DENSE}
    if set(by_name) != expected_modules:
        diff = set(by_name) ^ expected_modules
        raise W1FormatError(
            f"unexpected quantized-module inventory (symmetric difference: {sorted(diff)[:6]})"
        )
    if set(ckpt.dense_keys) != expected_dense:
        diff = set(ckpt.dense_keys) ^ expected_dense
        raise W1FormatError(
            f"unexpected dense-tensor inventory (symmetric difference: {sorted(diff)[:6]})"
        )
    head_dim = ckpt.tensor("model.layers.0.self_attn.q_norm.weight").shape[0]
    embed = by_name["model.embed_tokens"]
    q = by_name["model.layers.0.self_attn.q_proj"]
    k = by_name["model.layers.0.self_attn.k_proj"]
    return {
        "hidden_size": embed.in_features,
        "vocab_size": embed.out_features,
        "num_hidden_layers": len(layers),
        "intermediate_size": by_name["model.layers.0.mlp.gate_proj"].out_features,
        "head_dim": head_dim,
        "num_attention_heads": q.out_features // head_dim,
        "num_key_value_heads": k.out_features // head_dim,
    }


def hf_config(ckpt: W1Checkpoint, base: dict) -> dict:
    """HF ``config.json`` content for this checkpoint (without quantization settings)."""
    config = base.copy()
    for key, value in infer_dims(ckpt).items():
        if config.get(key) != value:
            raise W1FormatError(
                f"checkpoint implies {key}={value} but the base config has {config.get(key)!r}; "
                "use the config.json distributed with this model"
            )
    return config
