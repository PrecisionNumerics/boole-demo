"""A tiny Qwen3-shaped packed checkpoint that exercises the same code paths as the real one:
a tied embedding/head with *distinct* scales, negative and below-floor scales, fused-projection
shapes, and a vocabulary that is not a multiple of the kernels' tile (600)."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from boole_demo.qwen3 import QWEN3_1P7B_CONFIG

TINY_CONFIG = {
    **QWEN3_1P7B_CONFIG,
    "hidden_size": 256,
    "intermediate_size": 384,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "vocab_size": 600,
    "max_window_layers": 2,
    "max_position_embeddings": 512,
    "bos_token_id": 1,
    "eos_token_id": 2,
}
GROUP = 128


def _quantized(gen, n: int, k: int):
    codes = torch.randint(0, 256, (n, k // 8), generator=gen, dtype=torch.uint8)
    mags = torch.rand(n, k // GROUP, generator=gen) * 0.05 + 0.005
    mags[0, 0] = 1e-7  # below the 1e-5 floor
    signs = torch.where(torch.rand(n, k // GROUP, generator=gen) < 0.3, -1.0, 1.0)
    return codes, (mags * signs).to(torch.bfloat16)


def make_synthetic_checkpoint(path: Path, config: dict = TINY_CONFIG, seed: int = 0) -> None:
    gen = torch.Generator().manual_seed(seed)
    h, inter, hd = config["hidden_size"], config["intermediate_size"], config["head_dim"]
    q, kv = config["num_attention_heads"] * hd, config["num_key_value_heads"] * hd
    vocab = config["vocab_size"]
    tensors: dict[str, torch.Tensor] = {}

    def add(name: str, n: int, k: int) -> None:
        codes, scale = _quantized(gen, n, k)
        tensors[f"{name}.weight"] = codes
        tensors[f"{name}.weight_scale"] = scale

    add("model.embed_tokens", vocab, h)
    _, head_scale = _quantized(gen, vocab, h)
    tensors["lm_head.weight_scale"] = head_scale
    for i in range(config["num_hidden_layers"]):
        p = f"model.layers.{i}"
        add(f"{p}.self_attn.q_proj", q, h)
        add(f"{p}.self_attn.k_proj", kv, h)
        add(f"{p}.self_attn.v_proj", kv, h)
        add(f"{p}.self_attn.o_proj", h, q)
        add(f"{p}.mlp.gate_proj", inter, h)
        add(f"{p}.mlp.up_proj", inter, h)
        add(f"{p}.mlp.down_proj", h, inter)
        for norm, size in (
            ("input_layernorm", h),
            ("post_attention_layernorm", h),
            ("self_attn.q_norm", hd),
            ("self_attn.k_norm", hd),
        ):
            tensors[f"{p}.{norm}.weight"] = (1 + 0.1 * torch.randn(size, generator=gen)).to(
                torch.bfloat16
            )
    tensors["model.norm.weight"] = (1 + 0.1 * torch.randn(h, generator=gen)).to(torch.bfloat16)
    save_file(
        tensors,
        path,
        metadata={"format": "pt", "lm_head.weight": "model.embed_tokens.weight"},
    )
