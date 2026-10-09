import gguf
import numpy as np
import pytest
import torch

from boole_demo.gguf_export import (
    Q1_0_BLOCK_BYTES,
    Q1_0_GROUP,
    export_gguf,
    pack_q1_0,
    unpack_q1_0,
    verify_gguf,
)
from boole_demo.w1 import W1Checkpoint


def test_block_geometry_matches_gguf_py():
    assert gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q1_0] == (
        Q1_0_GROUP,
        Q1_0_BLOCK_BYTES,
    )


def test_block_layout_is_fp16_scale_then_sign_bytes(tiny_checkpoint):
    ckpt = W1Checkpoint(tiny_checkpoint)
    m = ckpt.module("model.layers.1.mlp.up_proj")
    codes, mags = ckpt.codes(m), ckpt.magnitudes(m)
    blocks = pack_q1_0(codes, mags)
    assert blocks.shape == (m.out_features, m.num_groups * Q1_0_BLOCK_BYTES)
    first = blocks[3, :Q1_0_BLOCK_BYTES]
    assert first[:2].tobytes() == np.float16(mags[3, 0].item()).tobytes()
    assert first[2:].tolist() == codes[3, :16].tolist()  # sign bytes copied untouched


def test_dequantised_blocks_equal_signed_fp16_magnitudes(tiny_checkpoint):
    ckpt = W1Checkpoint(tiny_checkpoint)
    for name in ("model.embed_tokens", "lm_head", "model.layers.0.mlp.down_proj"):
        m = ckpt.module(name)
        got = unpack_q1_0(
            pack_q1_0(ckpt.codes(m), ckpt.magnitudes(m)), m.out_features, m.in_features
        )
        expected = ckpt.dense(m).abs().half().float() * torch.sign(ckpt.dense(m))
        # Every value is exactly representable except those moved by the bf16 -> fp16 floor.
        close = torch.isclose(got, ckpt.dense(m), rtol=2e-3, atol=0)
        assert close.all()
        assert torch.equal(torch.sign(got), torch.sign(ckpt.dense(m)))
        assert torch.equal(got.abs(), expected.abs())


def test_shared_codes_keep_separate_scales(tiny_checkpoint):
    ckpt = W1Checkpoint(tiny_checkpoint)
    embed, head = ckpt.module("model.embed_tokens"), ckpt.module("lm_head")
    be = pack_q1_0(ckpt.codes(embed), ckpt.magnitudes(embed))
    bh = pack_q1_0(ckpt.codes(head), ckpt.magnitudes(head))
    sign_bytes = lambda b: b.reshape(b.shape[0], -1, Q1_0_BLOCK_BYTES)[..., 2:]  # noqa: E731
    scale_bytes = lambda b: b.reshape(b.shape[0], -1, Q1_0_BLOCK_BYTES)[..., :2]  # noqa: E731
    assert np.array_equal(sign_bytes(be), sign_bytes(bh))
    assert not np.array_equal(scale_bytes(be), scale_bytes(bh))


@pytest.mark.model
@pytest.mark.llama_cpp
def test_real_checkpoint_export_round_trips(real_checkpoint, tmp_path):
    import os
    from pathlib import Path

    llama_cpp = Path(os.environ.get("LLAMA_CPP_DIR", "third_party/llama.cpp"))
    if not (llama_cpp / "convert_hf_to_gguf.py").is_file():
        pytest.skip("llama.cpp checkout not found (scripts/setup_llama_cpp.sh)")
    ckpt = W1Checkpoint(real_checkpoint)
    out = export_gguf(
        ckpt,
        tmp_path / "m.gguf",
        llama_cpp,
        tokenizer=os.environ.get("BOOLE_TOKENIZER", "Qwen/Qwen3-1.7B"),
    )
    report = verify_gguf(ckpt, out)
    assert report["quantized"] == 198 and report["dense"] == 113
    assert report["max_scale_rel_err"] < 2e-3
