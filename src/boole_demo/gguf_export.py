"""Export the packed checkpoint as a llama.cpp GGUF using ``Q1_0`` for every 1-bit tensor.

llama.cpp's ``Q1_0`` block is 128 weights: one fp16 scale ``d`` followed by 16 bytes of sign
bits, LSB-first, bit 1 -> ``+d`` and bit 0 -> ``-d``. That is this checkpoint's own layout, so
the export is a lossless re-wrap: the sign bytes are copied unchanged and the only rounding is
bf16 -> fp16 on ``max(|scale|, eps)``.

The embedding and the output head are written as two tensors (``token_embd.weight`` and
``output.weight``) that carry identical sign bytes but their own scales.

Metadata (architecture hyper-parameters, tokenizer, chat template) is produced by llama.cpp's
own ``convert_hf_to_gguf.py --vocab-only`` and copied verbatim, so it always matches what the
pinned llama.cpp would write for a dense Qwen3.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter

from .model_dir import DEFAULT_TOKENIZER_REPO, copy_tokenizer_files
from .qwen3 import hf_config
from .w1 import EMBEDDING, LM_HEAD, W1Checkpoint, W1FormatError, unpack_signs

Q1_0_GROUP = 128
Q1_0_BLOCK_BYTES = 18
_ARCH = gguf.MODEL_ARCH.QWEN3


def pack_q1_0(codes: torch.Tensor, magnitudes: torch.Tensor) -> np.ndarray:
    """Pack uint8 ``[N, K/8]`` codes and float32 ``[N, K/128]`` magnitudes into Q1_0 blocks."""
    if sys.byteorder != "little":
        raise RuntimeError("GGUF Q1_0 export assumes a little-endian host")
    n, groups = magnitudes.shape
    if codes.shape != (n, groups * 16):
        raise W1FormatError(f"codes {tuple(codes.shape)} do not match {groups} groups of 128")
    scale_bytes = magnitudes.to(torch.float16).contiguous().view(torch.uint8).view(n, groups, 2)
    blocks = torch.cat([scale_bytes, codes.view(n, groups, 16)], dim=-1)
    return blocks.reshape(n, groups * Q1_0_BLOCK_BYTES).numpy()


def unpack_q1_0(blocks: np.ndarray, rows: int, cols: int) -> torch.Tensor:
    """Independent float32 dequantisation of ``Q1_0`` rows (mirrors ggml's reference loop)."""
    groups = cols // Q1_0_GROUP
    raw = torch.from_numpy(np.ascontiguousarray(blocks)).view(rows, groups, Q1_0_BLOCK_BYTES)
    d = raw[..., :2].contiguous().view(torch.float16).float()  # [rows, groups, 1]
    signs = unpack_signs(raw[..., 2:].reshape(rows, groups * 16)).float()
    return (signs.view(rows, groups, Q1_0_GROUP) * d).reshape(rows, cols)


def _vocab_metadata(llama_cpp: Path, model_dir: Path, out: Path) -> GGUFReader:
    script = llama_cpp / "convert_hf_to_gguf.py"
    if not script.is_file():
        raise FileNotFoundError(f"{script} not found; run scripts/setup_llama_cpp.sh")
    subprocess.run(
        [sys.executable, str(script), str(model_dir), "--vocab-only", "--outfile", str(out)],
        check=True,
    )
    return GGUFReader(out)


def _copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    skipped = {"general.architecture", "general.file_type", "general.quantization_version"}
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key in skipped:
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        writer.add_key_value(key, field.contents(), vtype, sub_type=sub_type)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q1_0)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)


def export_gguf(
    ckpt: W1Checkpoint,
    out_path: Path,
    llama_cpp: Path,
    *,
    tokenizer: str | Path = DEFAULT_TOKENIZER_REPO,
    base_config: dict | None = None,
) -> Path:
    if ckpt.group_size != Q1_0_GROUP:
        raise W1FormatError(
            f"Q1_0 needs group size {Q1_0_GROUP}; this checkpoint uses {ckpt.group_size}"
        )
    config = hf_config(ckpt, base_config)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        stub = Path(tmp) / "stub"
        copy_tokenizer_files(stub, tokenizer)
        (stub / "config.json").write_text(json.dumps(config, indent=2))
        reader = _vocab_metadata(llama_cpp, stub, Path(tmp) / "vocab.gguf")

        arch = reader.get_field("general.architecture").contents()
        if arch != gguf.MODEL_ARCH_NAMES[_ARCH]:
            raise W1FormatError(
                f"llama.cpp converter produced architecture {arch!r}, expected qwen3"
            )
        writer = GGUFWriter(out_path, arch)  # also writes general.architecture
        _copy_metadata(reader, writer)

        names = gguf.get_tensor_name_map(_ARCH, config["num_hidden_layers"])

        def gguf_name(hf_name: str) -> str:
            mapped = names.get_name(hf_name, try_suffixes=(".weight", ".bias"))
            if mapped is None:
                raise W1FormatError(f"no GGUF tensor name for {hf_name!r}")
            return mapped

        for module in ckpt.modules:
            blocks = pack_q1_0(ckpt.codes(module), ckpt.magnitudes(module))
            # For uint8 data gguf-py takes the *byte* shape [N, K/128 * 18] and derives [N, K].
            writer.add_tensor(
                gguf_name(f"{module.name}.weight"), blocks, raw_dtype=GGMLQuantizationType.Q1_0
            )
        for key in ckpt.dense_keys:
            writer.add_tensor(gguf_name(key), ckpt.tensor(key).float().numpy())

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
    return out_path


def verify_gguf(ckpt: W1Checkpoint, path: Path, *, rows_per_check: int = 4096) -> dict:
    """Re-read ``path`` and check every tensor against the source checkpoint.

    Each quantized tensor must dequantise to ``sign * fp16(max(|scale|, eps))`` exactly, and
    each dense tensor must equal the source norm weights as float32.
    """
    reader = GGUFReader(path)
    by_name = {t.name: t for t in reader.tensors}
    names = gguf.get_tensor_name_map(_ARCH, int(reader.get_field("qwen3.block_count").contents()))
    checked = {"quantized": 0, "dense": 0, "max_scale_rel_err": 0.0}
    for module in ckpt.modules:
        tensor = by_name.pop(names.get_name(f"{module.name}.weight", try_suffixes=(".weight",)))
        if tensor.tensor_type != GGMLQuantizationType.Q1_0:
            raise AssertionError(f"{tensor.name}: expected Q1_0, found {tensor.tensor_type.name}")
        if tuple(int(x) for x in tensor.shape) != (module.in_features, module.out_features):
            raise AssertionError(f"{tensor.name}: shape {tensor.shape}")
        codes, mags = ckpt.codes(module), ckpt.magnitudes(module)
        data = np.asarray(tensor.data).reshape(module.out_features, -1)
        for start in range(0, module.out_features, rows_per_check):
            sl = slice(start, start + rows_per_check)
            got = unpack_q1_0(data[sl], data[sl].shape[0], module.in_features)
            expected = (
                unpack_signs(codes[sl]).float().view(-1, module.num_groups, module.group_size)
                * mags[sl].to(torch.float16).float().unsqueeze(-1)
            ).reshape(got.shape)
            if not torch.equal(got, expected):
                raise AssertionError(f"{tensor.name}: rows {sl} differ from the source")
        rel = ((mags.to(torch.float16).float() - mags).abs() / mags).max().item()
        checked["max_scale_rel_err"] = max(checked["max_scale_rel_err"], rel)
        checked["quantized"] += 1
    for key in ckpt.dense_keys:
        tensor = by_name.pop(names.get_name(key, try_suffixes=(".weight",)))
        if not np.array_equal(
            np.asarray(tensor.data).reshape(-1), ckpt.tensor(key).float().numpy().reshape(-1)
        ):
            raise AssertionError(f"{tensor.name}: dense tensor differs from the source")
        checked["dense"] += 1
    if by_name:
        raise AssertionError(f"unexpected extra tensors in GGUF: {sorted(by_name)[:5]}")
    kinds = {m.kind for m in ckpt.modules}
    if not {EMBEDDING, LM_HEAD} <= kinds:
        raise AssertionError("checkpoint lacks an embedding/head pair")
    return checked
