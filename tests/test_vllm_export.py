import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from boole_demo.vllm_export import ZERO_POINT, export_vllm_dir, humming_tensors
from boole_demo.w1 import SCALE_EPS, W1Checkpoint, unpack_signs


def test_tensor_inventory_and_dtypes(tiny_checkpoint):
    ckpt = W1Checkpoint(tiny_checkpoint)
    tensors = humming_tensors(ckpt)
    assert "lm_head.weight" not in tensors  # tied to the embedding's codes by the plugin
    for module in ckpt.modules:
        if module.kind != "lm_head":
            w = tensors[f"{module.name}.weight"]
            assert w.dtype == torch.int32 and w.shape == (
                module.out_features,
                module.in_features // 32,
            )
        s = tensors[f"{module.name}.weight_scale"]
        assert s.dtype == torch.bfloat16 and s.shape == (module.out_features, module.num_groups)
        assert (s.float() >= 2 * SCALE_EPS * 0.99).all()
    assert not [k for k in tensors if k.endswith("zero_point")]  # the constant is not stored
    assert not torch.equal(
        tensors["lm_head.weight_scale"], tensors["model.embed_tokens.weight_scale"]
    )


def test_int32_words_are_the_source_bytes(tiny_checkpoint):
    ckpt = W1Checkpoint(tiny_checkpoint)
    m = ckpt.module("model.layers.0.self_attn.q_proj")
    words = humming_tensors(ckpt)[f"{m.name}.weight"]
    # bit j of word w is weight 32*w + j
    bits = (words.unsqueeze(-1) >> torch.arange(32, dtype=torch.int32)) & 1
    assert torch.equal(
        bits.reshape(m.out_features, -1).float() * 2 - 1, unpack_signs(ckpt.codes(m)).float()
    )


def test_affine_form_reproduces_signed_magnitudes(tiny_checkpoint):
    ckpt = W1Checkpoint(tiny_checkpoint)
    m = ckpt.module("model.layers.1.mlp.down_proj")
    t = humming_tensors(ckpt)
    bits = (
        (t[f"{m.name}.weight"].unsqueeze(-1) >> torch.arange(32, dtype=torch.int32)) & 1
    ).reshape(m.out_features, -1)
    scale = t[f"{m.name}.weight_scale"].float().repeat_interleave(m.group_size, dim=1)
    assert torch.equal(
        (bits.float() - ZERO_POINT) * scale, ckpt.dense(m).to(torch.bfloat16).float()
    )


def test_exported_directory(tiny_checkpoint, tiny_config, tmp_path):
    out = export_vllm_dir(
        W1Checkpoint(tiny_checkpoint), tmp_path / "m", tokenizer=None, base_config=tiny_config
    )
    config = json.loads((out / "config.json").read_text())
    assert config["tie_word_embeddings"] is True
    assert config["quantization_config"]["quant_method"] == "w1_humming"
    assert config["quantization_config"]["dtype"] == "uint1"
    assert config["quantization_config"]["zero_point"] == ZERO_POINT
    assert set(load_file(out / "model.safetensors")) == set(
        humming_tensors(W1Checkpoint(tiny_checkpoint))
    )
    assert json.loads((out / "boole_export.json").read_text())["source_sha256"]


@pytest.mark.gpu
def test_humming_reference_dequantiser_agrees(tiny_checkpoint):
    """Humming's own dequantiser must read our tensors as exactly +/- the effective scales."""
    from humming import dtypes
    from humming.utils.weight import dequantize_weight

    ckpt = W1Checkpoint(tiny_checkpoint)
    tensors = humming_tensors(ckpt)
    for name in (
        "model.layers.0.mlp.gate_proj",
        "model.layers.1.self_attn.k_proj",
        "model.embed_tokens",
    ):
        m = ckpt.module(name)
        got = dequantize_weight(
            tensors[f"{name}.weight"].cuda(),
            weight_scale=tensors[f"{name}.weight_scale"].cuda(),
            zero_point=torch.full_like(tensors[f"{name}.weight_scale"], ZERO_POINT).cuda(),
            weight_scale_2=None,
            dtype=dtypes.DataType.from_str("uint1"),
            packed=True,
        )
        assert torch.equal(got.cpu().to(torch.bfloat16), ckpt.dense(m, torch.bfloat16))


_ITEMSIZE = {"U8": 1, "I32": 4, "BF16": 2, "F16": 2, "F32": 4}


def _tensor_bytes(path) -> int:
    total = 0
    with safe_open(path, "pt") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open is not a Mapping
            piece = f.get_slice(key)
            count = 1
            for dim in piece.get_shape():
                count *= dim
            total += count * _ITEMSIZE[piece.get_dtype()]
    return total


def test_export_adds_no_bytes_over_the_source(tiny_checkpoint, tiny_config, tmp_path):
    """The 1-bit model must not grow: exported tensors total exactly the source's bytes."""
    out = export_vllm_dir(
        W1Checkpoint(tiny_checkpoint), tmp_path / "m", tokenizer=None, base_config=tiny_config
    )
    assert _tensor_bytes(out / "model.safetensors") == _tensor_bytes(tiny_checkpoint)
