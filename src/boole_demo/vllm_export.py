"""Export the packed checkpoint as a vLLM model directory served with Humming kernels.

Every 1-bit tensor is written in Humming's native ``uint1`` layout with an fp zero point, which
turns ``W = +/-m`` into ``(q - 0.5) * (2 * m)``:

    <name>.weight        int32 [N, K/32]   the source sign bytes, reinterpreted (LSB-first)
    <name>.weight_scale  bf16  [N, K/G]    2 * max(|scale|, eps)

so no sign bit is re-ordered and Humming/vLLM apply no further transform to the scales. The zero
point is the constant 0.5 for every group, so it is *not stored*: it is declared once in
``quantization_config`` and supplied by the plugin (storing it would add a constant tensor as large
as all the scales). The tied embedding keeps *one* ``model.embed_tokens.weight`` and two scale
tensors (``model.embed_tokens.weight_scale`` and ``lm_head.weight_scale``); ``lm_head.weight`` is
deliberately absent and is tied to the embedding's codes by the plugin in
:mod:`boole_demo.vllm_quant`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from . import __version__
from .model_dir import DEFAULT_TOKENIZER_REPO, copy_tokenizer_files
from .qwen3 import hf_config
from .w1 import LM_HEAD, SCALE_EPS, W1Checkpoint

QUANT_METHOD = "w1_humming"
ZERO_POINT = 0.5


def quantization_config(group_size: int) -> dict:
    return {
        "quant_method": QUANT_METHOD,
        "dtype": "uint1",
        "group_size": group_size,
        "has_zero_point": True,
        "is_fp_zero_point": True,
        "zero_point": ZERO_POINT,
        "scale_floor": SCALE_EPS,
    }


def humming_tensors(ckpt: W1Checkpoint) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for module in ckpt.modules:
        if module.kind != LM_HEAD:
            codes = ckpt.codes(module)
            if codes.shape[1] % 4:
                raise ValueError(f"{module.name}: K must be a multiple of 32 for int32 packing")
            tensors[f"{module.name}.weight"] = codes.view(torch.int32)
        tensors[f"{module.name}.weight_scale"] = (2.0 * ckpt.magnitudes(module)).to(torch.bfloat16)
    for key in ckpt.dense_keys:
        tensors[key] = ckpt.tensor(key).to(torch.bfloat16)
    return tensors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_vllm_dir(
    ckpt: W1Checkpoint,
    out_dir: Path,
    *,
    tokenizer: str | Path | None = DEFAULT_TOKENIZER_REPO,
    base_config: dict | None = None,
) -> Path:
    """Write the model directory. ``tokenizer=None`` omits tokenizer files (token-id use only)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = hf_config(ckpt, base_config)
    config["quantization_config"] = quantization_config(ckpt.group_size)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if tokenizer is not None:
        copy_tokenizer_files(out_dir, tokenizer)
    save_file(humming_tensors(ckpt), out_dir / "model.safetensors", metadata={"format": "pt"})
    (out_dir / "boole_export.json").write_text(
        json.dumps(
            {
                "tool": f"boole-demo {__version__}",
                "source": ckpt.path.name,
                "source_sha256": _sha256(ckpt.path),
                "scale_transform": f"max(abs(scale), {SCALE_EPS})",
                "tied_embedding": "model.embed_tokens.weight shared with lm_head; scales distinct",
            },
            indent=2,
        )
        + "\n"
    )
    return out_dir
