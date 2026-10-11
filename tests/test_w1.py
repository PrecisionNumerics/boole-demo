import pytest
import torch

from boole_demo.qwen3 import hf_config, infer_dims
from boole_demo.w1 import SCALE_EPS, W1Checkpoint, W1FormatError

from .synthetic import synthetic_tensors, write_checkpoint


def test_discovery_and_shared_embedding(tiny_checkpoint, tiny_config):
    ckpt = W1Checkpoint(tiny_checkpoint)
    by_name = {m.name: m for m in ckpt.modules}
    assert len(ckpt.modules) == 2 + 7 * tiny_config["num_hidden_layers"]
    assert ckpt.group_size == 128
    head, embed = by_name["lm_head"], by_name["model.embed_tokens"]
    assert head.code_key == embed.code_key == "model.embed_tokens.weight"
    assert torch.equal(ckpt.codes(head), ckpt.codes(embed))
    assert head.scale_key != embed.scale_key
    assert not torch.equal(ckpt.tensor(head.scale_key), ckpt.tensor(embed.scale_key))
    assert "model.norm.weight" in ckpt.dense_keys


def test_magnitudes_apply_abs_and_floor(tiny_checkpoint):
    ckpt = W1Checkpoint(tiny_checkpoint)
    module = next(m for m in ckpt.modules if m.name == "model.layers.0.mlp.down_proj")
    raw, magnitudes = ckpt.tensor(module.scale_key).float(), ckpt.magnitudes(module)
    assert (raw < 0).any() and (raw.abs() < SCALE_EPS).any()
    assert (magnitudes >= SCALE_EPS).all()
    large = raw.abs() >= SCALE_EPS
    assert torch.equal(magnitudes[large], raw.abs()[large])


def test_dimensions_are_inferred_and_checked(tiny_checkpoint, tiny_config):
    ckpt = W1Checkpoint(tiny_checkpoint)
    dims = infer_dims(ckpt)
    assert dims["vocab_size"] == 600 and dims["num_hidden_layers"] == 2
    assert hf_config(ckpt, tiny_config)["tie_word_embeddings"] is True
    with pytest.raises(W1FormatError, match="hidden_size"):
        hf_config(ckpt, {**tiny_config, "hidden_size": 512})


def test_missing_codes_are_rejected(tmp_path):
    tensors = synthetic_tensors()
    del tensors["model.layers.0.mlp.up_proj.weight"]
    with pytest.raises(W1FormatError, match="no packed codes"):
        W1Checkpoint(write_checkpoint(tmp_path / "broken.safetensors", tensors))


def test_mixed_group_sizes_are_rejected(tmp_path):
    tensors = synthetic_tensors()
    key = "model.layers.1.mlp.down_proj.weight_scale"
    tensors[key] = tensors[key].repeat_interleave(2, dim=1)
    with pytest.raises(W1FormatError, match="mixed group sizes"):
        W1Checkpoint(write_checkpoint(tmp_path / "mixed.safetensors", tensors))


def test_unexpected_tensors_are_rejected(tmp_path):
    tensors = synthetic_tensors()
    tensors["model.layers.0.self_attn.extra.weight"] = torch.zeros(4, dtype=torch.bfloat16)
    ckpt = W1Checkpoint(write_checkpoint(tmp_path / "extra.safetensors", tensors))
    with pytest.raises(W1FormatError, match="dense-tensor inventory"):
        infer_dims(ckpt)
