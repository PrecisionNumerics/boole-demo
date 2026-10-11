import json

import pytest
import torch
from safetensors.torch import load_file

from boole_demo import QUANT_METHOD, ZERO_POINT
from boole_demo.cli import main
from boole_demo.vllm_export import export_vllm_dir, humming_tensors, quantization_config
from boole_demo.w1 import SCALE_EPS, W1Checkpoint

from .synthetic import TINY_CONFIG, write_checkpoint


def _signs(codes: torch.Tensor) -> torch.Tensor:
    """Sign bits LSB-first: bit 1 is +1, bit 0 is -1."""
    bits = (codes.unsqueeze(-1) >> torch.arange(8, dtype=torch.uint8)) & 1
    return bits.reshape(codes.shape[0], -1).float() * 2 - 1


def test_tensor_inventory_and_dtypes(tiny_checkpoint):
    ckpt = W1Checkpoint(tiny_checkpoint)
    tensors = humming_tensors(ckpt)
    assert "lm_head.weight" not in tensors  # tied to the embedding's codes by the plugin
    for module in ckpt.modules:
        if module.name != "lm_head":
            weight = tensors[f"{module.name}.weight"]
            assert weight.dtype == torch.int32
            assert weight.shape == (module.out_features, module.in_features // 32)
        scale = tensors[f"{module.name}.weight_scale"]
        assert scale.dtype == torch.bfloat16
        assert scale.shape == (module.out_features, module.in_features // ckpt.group_size)
        assert (scale.float() >= 2 * SCALE_EPS * 0.99).all()
    assert not [key for key in tensors if key.endswith("zero_point")]  # the constant is not stored
    assert not torch.equal(
        tensors["lm_head.weight_scale"], tensors["model.embed_tokens.weight_scale"]
    )


def test_int32_words_are_the_source_bytes(tiny_checkpoint):
    ckpt = W1Checkpoint(tiny_checkpoint)
    module = next(m for m in ckpt.modules if m.name == "model.layers.0.self_attn.q_proj")
    words = humming_tensors(ckpt)[f"{module.name}.weight"]
    # bit j of word w is weight 32 * w + j
    bits = (words.unsqueeze(-1) >> torch.arange(32, dtype=torch.int32)) & 1
    assert torch.equal(
        bits.reshape(module.out_features, -1).float() * 2 - 1, _signs(ckpt.codes(module))
    )


def test_affine_form_reproduces_signed_magnitudes(tiny_checkpoint):
    ckpt = W1Checkpoint(tiny_checkpoint)
    module = next(m for m in ckpt.modules if m.name == "model.layers.1.mlp.down_proj")
    tensors = humming_tensors(ckpt)
    words, scale = tensors[f"{module.name}.weight"], tensors[f"{module.name}.weight_scale"].float()
    bits = ((words.unsqueeze(-1) >> torch.arange(32, dtype=torch.int32)) & 1).reshape(
        module.out_features, -1
    )
    groups = scale.repeat_interleave(ckpt.group_size, dim=1)
    dequantized = (bits.float() - ZERO_POINT) * groups
    expected = _signs(ckpt.codes(module)) * ckpt.magnitudes(module).repeat_interleave(
        ckpt.group_size, dim=1
    )
    torch.testing.assert_close(dequantized, expected, rtol=1e-2, atol=1e-6)


def test_export_directory(tiny_checkpoint, tiny_config, tmp_path):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (tokenizer / name).write_text("{}")
    with W1Checkpoint(tiny_checkpoint) as ckpt:
        out = export_vllm_dir(ckpt, tmp_path / "out", tokenizer=tokenizer, base_config=tiny_config)
    config = json.loads((out / "config.json").read_text())
    assert config["quantization_config"] == quantization_config(128)
    assert config["quantization_config"]["quant_method"] == QUANT_METHOD
    assert (out / "tokenizer.json").is_file() and (out / "tokenizer_config.json").is_file()
    exported = load_file(out / "model.safetensors")
    assert "model.embed_tokens.weight" in exported and "lm_head.weight" not in exported


def test_missing_tokenizer_files_are_reported(tiny_checkpoint, tiny_config, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with W1Checkpoint(tiny_checkpoint) as ckpt, pytest.raises(FileNotFoundError):
        export_vllm_dir(ckpt, tmp_path / "out", tokenizer=empty, base_config=tiny_config)


def test_cli_exports_a_local_model_directory(tmp_path, capsys):
    model = tmp_path / "model"
    model.mkdir()
    write_checkpoint(model / "Boole-tiny.safetensors")
    (model / "config.json").write_text(json.dumps(TINY_CONFIG))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (model / name).write_text("{}")
    main(["vllm", str(model), "--out", str(tmp_path / "out")])
    assert "Ready for vLLM + Humming" in capsys.readouterr().out
    assert (tmp_path / "out" / "model.safetensors").is_file()
