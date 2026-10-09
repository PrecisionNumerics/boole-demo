import pytest
import torch
from safetensors.torch import load_file, save_file

from boole_demo.qwen3 import hf_config, infer_dims
from boole_demo.w1 import (
    SCALE_EPS,
    W1Checkpoint,
    W1FormatError,
    pack_signs,
    unpack_signs,
)


def test_all_byte_values_round_trip_lsb_first():
    codes = torch.arange(256, dtype=torch.uint8).view(2, 128)
    signs = unpack_signs(codes)
    assert signs.shape == (2, 1024)
    # byte 0b00000001 -> first element +1, the other seven -1 (LSB-first)
    assert signs[0, :8].tolist() == [-1] * 8
    assert signs[0, 8:16].tolist() == [1, -1, -1, -1, -1, -1, -1, -1]
    assert torch.equal(pack_signs(signs), codes)


def test_discovery_and_shared_embedding(tiny_checkpoint, tiny_config):
    ckpt = W1Checkpoint(tiny_checkpoint)
    kinds = {m.name: m.kind for m in ckpt.modules}
    assert kinds["model.embed_tokens"] == "embedding" and kinds["lm_head"] == "lm_head"
    assert len(ckpt.modules) == 2 + 7 * tiny_config["num_hidden_layers"]
    assert ckpt.group_size == 128
    head, embed = ckpt.module("lm_head"), ckpt.module("model.embed_tokens")
    assert head.code_key == embed.code_key == "model.embed_tokens.weight"
    assert torch.equal(ckpt.codes(head), ckpt.codes(embed))
    assert not torch.equal(ckpt.raw_scales(head), ckpt.raw_scales(embed))
    assert infer_dims(ckpt)["vocab_size"] == 600
    assert hf_config(ckpt, tiny_config)["tie_word_embeddings"] is True


def test_magnitudes_apply_abs_and_floor(tiny_checkpoint):
    ckpt = W1Checkpoint(tiny_checkpoint)
    m = ckpt.module("model.layers.0.mlp.down_proj")
    raw, mags = ckpt.raw_scales(m), ckpt.magnitudes(m)
    assert (raw < 0).any() and (raw.abs() < SCALE_EPS).any()
    assert (mags >= SCALE_EPS).all()
    big = raw.abs() >= SCALE_EPS
    assert torch.equal(mags[big], raw.abs()[big])
    dense = ckpt.dense(m)
    assert dense.abs().min().item() == pytest.approx(SCALE_EPS)
    # signs come from the codes only, never from the sign of the stored scale
    assert torch.equal(torch.sign(dense), unpack_signs(ckpt.codes(m)).float())


def test_missing_alias_is_rejected(tiny_checkpoint, tmp_path):
    tensors = load_file(tiny_checkpoint)
    path = tmp_path / "no-alias.safetensors"
    save_file(tensors, path, metadata={"format": "pt"})
    with pytest.raises(W1FormatError, match="no packed codes"):
        W1Checkpoint(path)


def test_mixed_group_sizes_are_rejected(tiny_checkpoint, tmp_path):
    tensors = load_file(tiny_checkpoint)
    name = "model.layers.0.mlp.down_proj.weight_scale"
    tensors[name] = tensors[name][:, :1].contiguous()  # 384 columns / 1 group = 384
    path = tmp_path / "mixed.safetensors"
    save_file(
        tensors, path, metadata={"format": "pt", "lm_head.weight": "model.embed_tokens.weight"}
    )
    with pytest.raises(W1FormatError, match="mixed group sizes"):
        W1Checkpoint(path)


def test_config_mismatch_is_rejected(tiny_checkpoint):
    with pytest.raises(W1FormatError, match="--base-config"):
        hf_config(W1Checkpoint(tiny_checkpoint))  # default Qwen3-1.7B dims do not fit
