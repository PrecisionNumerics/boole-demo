"""Qwen3 architecture facts for this checkpoint.

The safetensors file carries no config, so dimensions are inferred from tensor shapes and
the remaining constants (RoPE base, norm epsilon, ...) come from the public
``Qwen/Qwen3-1.7B`` config, which the shapes match exactly.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .w1 import W1Checkpoint, W1FormatError

# Verbatim from https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json
QWEN3_1P7B_CONFIG: dict = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "max_position_embeddings": 40960,
    "max_window_layers": 28,
    "model_type": "qwen3",
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}

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


def hf_config(ckpt: W1Checkpoint, base: dict | None = None) -> dict:
    """HF ``config.json`` content for this checkpoint (without quantization settings)."""
    config = copy.deepcopy(base if base is not None else QWEN3_1P7B_CONFIG)
    for key, value in infer_dims(ckpt).items():
        if config.get(key) != value:
            raise W1FormatError(
                f"checkpoint implies {key}={value} but the base config has {config.get(key)!r}; "
                "pass --base-config with the config this model was trained from"
            )
    return config


def load_base_config(path: str | Path | None) -> dict | None:
    return None if path is None else json.loads(Path(path).read_text())
