"""A tiny Qwen3-shaped packed checkpoint that exercises the same code paths as the real one:
a tied embedding and head with *distinct* scales, negative and below-floor scales, and a
vocabulary that is not a multiple of the kernels' tile."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

GROUP = 128

TINY_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 256,
    "intermediate_size": 384,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "vocab_size": 600,
    "tie_word_embeddings": True,
    "max_position_embeddings": 512,
}

LAYER_LINEARS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _quantized(gen: torch.Generator, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    codes = torch.randint(0, 256, (n, k // 8), generator=gen, dtype=torch.uint8)
    mags = torch.rand(n, k // GROUP, generator=gen) * 0.05 + 0.005
    mags[0, 0] = 1e-7  # below the scale floor
    signs = torch.where(torch.rand(n, k // GROUP, generator=gen) < 0.3, -1.0, 1.0)
    return codes, (mags * signs).to(torch.bfloat16)


def synthetic_tensors(config: dict = TINY_CONFIG, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    h, inter, hd = config["hidden_size"], config["intermediate_size"], config["head_dim"]
    q, kv = config["num_attention_heads"] * hd, config["num_key_value_heads"] * hd
    shapes = {
        "self_attn.q_proj": (q, h),
        "self_attn.k_proj": (kv, h),
        "self_attn.v_proj": (kv, h),
        "self_attn.o_proj": (h, q),
        "mlp.gate_proj": (inter, h),
        "mlp.up_proj": (inter, h),
        "mlp.down_proj": (h, inter),
    }
    tensors: dict[str, torch.Tensor] = {}

    def add(name: str, n: int, k: int) -> None:
        codes, scale = _quantized(gen, n, k)
        tensors[f"{name}.weight"] = codes
        tensors[f"{name}.weight_scale"] = scale

    add("model.embed_tokens", config["vocab_size"], h)
    tensors["lm_head.weight_scale"] = _quantized(gen, config["vocab_size"], h)[1]
    for i in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{i}"
        for projection in LAYER_LINEARS:
            add(f"{prefix}.{projection}", *shapes[projection])
        for norm, size in (
            ("input_layernorm", h),
            ("post_attention_layernorm", h),
            ("self_attn.q_norm", hd),
            ("self_attn.k_norm", hd),
        ):
            tensors[f"{prefix}.{norm}.weight"] = (1 + 0.1 * torch.randn(size, generator=gen)).to(
                torch.bfloat16
            )
    tensors["model.norm.weight"] = (1 + 0.1 * torch.randn(h, generator=gen)).to(torch.bfloat16)
    return tensors


def write_checkpoint(path: Path, tensors: dict[str, torch.Tensor] | None = None) -> Path:
    """Write a packed checkpoint; ``lm_head.weight`` aliases the embedding's codes."""
    save_file(
        tensors if tensors is not None else synthetic_tensors(),
        path,
        metadata={"format": "pt", "lm_head.weight": "model.embed_tokens.weight"},
    )
    return path
