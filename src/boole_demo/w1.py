"""Reader and reference decoder for the packed 1-bit ("W1") checkpoint.

For every quantized module ``<name>`` the file holds::

    <name>.weight        uint8  [N, K/8]   sign bits, LSB-first: bit 1 -> +1, bit 0 -> -1
    <name>.weight_scale  bf16   [N, K/G]   learned per-group scale, any sign

and the effective weight is ``W[n, k] = sign[n, k] * max(|scale[n, k // G]|, SCALE_EPS)``.
The sign of a stored scale carries no information; ``abs`` and the floor are part of the
format, so every consumer in this repository works on :meth:`W1Checkpoint.magnitudes`.

The input embedding and the output head share one code tensor (``lm_head.weight`` is
listed in the safetensors ``__metadata__`` as an alias of ``model.embed_tokens.weight``)
but have *separate* scale tensors, so they are two modules over identical signs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

SCALE_EPS = 1e-5

EMBEDDING = "embedding"
LM_HEAD = "lm_head"
LINEAR = "linear"


class W1FormatError(ValueError):
    """The file does not follow the packed 1-bit layout described above."""


@dataclass(frozen=True)
class W1Module:
    name: str
    kind: str
    out_features: int
    in_features: int
    group_size: int
    code_key: str
    scale_key: str

    @property
    def num_groups(self) -> int:
        return self.in_features // self.group_size


def unpack_signs(codes: torch.Tensor) -> torch.Tensor:
    """uint8 ``[N, K/8]`` LSB-first codes to int8 ``[N, K]`` of +1/-1."""
    shifts = torch.arange(8, device=codes.device, dtype=torch.uint8)
    bits = (codes.unsqueeze(-1) >> shifts) & 1
    return (bits.reshape(codes.shape[0], -1).to(torch.int8) * 2) - 1


def pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`unpack_signs` (any value >= 0 is a one bit)."""
    n, k = signs.shape
    if k % 8:
        raise W1FormatError(f"cannot pack {k} columns into whole bytes")
    bits = (signs >= 0).reshape(n, k // 8, 8).to(torch.uint8)
    shifts = torch.arange(8, device=signs.device, dtype=torch.uint8)
    return (bits << shifts).sum(dim=-1, dtype=torch.uint8)


class W1Checkpoint:
    """A validated view of the packed safetensors file (tensors are read lazily)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = safe_open(self.path, framework="pt", device="cpu")
        self.metadata: dict[str, str] = dict(self._file.metadata() or {})
        self._keys = set(self._file.keys())
        self.modules: tuple[W1Module, ...] = self._discover()
        claimed = {m.code_key for m in self.modules} | {m.scale_key for m in self.modules}
        self.dense_keys: tuple[str, ...] = tuple(sorted(self._keys - claimed))

    def close(self) -> None:
        self._file = None  # safe_open handles are released with the object

    def __enter__(self) -> W1Checkpoint:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._file.get_slice(key).get_shape())

    def _dtype(self, key: str) -> str:
        return self._file.get_slice(key).get_dtype()

    def _discover(self) -> tuple[W1Module, ...]:
        suffix = ".weight_scale"
        modules = []
        for scale_key in sorted(k for k in self._keys if k.endswith(suffix)):
            name = scale_key[: -len(suffix)]
            weight_key = f"{name}.weight"
            code_key = weight_key if weight_key in self._keys else self.metadata.get(weight_key)
            if code_key is None or code_key not in self._keys:
                raise W1FormatError(
                    f"{scale_key!r} has no packed codes: {weight_key!r} is neither a tensor nor "
                    "an alias declared in the safetensors metadata"
                )
            if self._dtype(code_key) != "U8" or len(self._shape(code_key)) != 2:
                raise W1FormatError(f"{code_key!r} must be a 2-D uint8 tensor")
            if len(self._shape(scale_key)) != 2:
                raise W1FormatError(f"{scale_key!r} must be a 2-D tensor")
            if self._dtype(scale_key) not in ("BF16", "F16", "F32"):
                raise W1FormatError(f"{scale_key!r} must be a floating-point tensor")
            rows, code_cols = self._shape(code_key)
            scale_rows, groups = self._shape(scale_key)
            in_features = code_cols * 8
            if scale_rows != rows or in_features % groups:
                raise W1FormatError(
                    f"{name!r}: scales {self._shape(scale_key)} do not tile codes "
                    f"{self._shape(code_key)}"
                )
            if name.endswith("embed_tokens"):
                kind = EMBEDDING
            elif name == "lm_head":
                kind = LM_HEAD
            else:
                kind = LINEAR
            if code_key != weight_key and kind != LM_HEAD:
                raise W1FormatError(f"{name!r}: only the output head may alias another module")
            modules.append(
                W1Module(
                    name=name,
                    kind=kind,
                    out_features=rows,
                    in_features=in_features,
                    group_size=in_features // groups,
                    code_key=code_key,
                    scale_key=scale_key,
                )
            )
        if not modules:
            raise W1FormatError("no '<name>.weight_scale' tensors found")
        if len({m.group_size for m in modules}) != 1:
            raise W1FormatError(f"mixed group sizes: { {m.name: m.group_size for m in modules} }")
        return tuple(modules)

    @property
    def group_size(self) -> int:
        return self.modules[0].group_size

    def module(self, name: str) -> W1Module:
        for m in self.modules:
            if m.name == name:
                return m
        raise KeyError(name)

    def tensor(self, key: str) -> torch.Tensor:
        return self._file.get_tensor(key)

    def codes(self, module: W1Module) -> torch.Tensor:
        """Packed uint8 signs ``[N, K/8]``; shared by the embedding and the head."""
        return self.tensor(module.code_key).contiguous()

    def raw_scales(self, module: W1Module) -> torch.Tensor:
        return self.tensor(module.scale_key).float()

    def magnitudes(self, module: W1Module) -> torch.Tensor:
        """Effective per-group magnitudes ``max(|scale|, SCALE_EPS)`` as float32 ``[N, G]``."""
        return self.raw_scales(module).abs().clamp_min(SCALE_EPS)

    def dense(
        self,
        module: W1Module,
        dtype: torch.dtype = torch.float32,
        rows: slice | None = None,
        *,
        signed_scales: bool = False,
    ) -> torch.Tensor:
        """Dequantized ``[N, K]`` weight. ``signed_scales`` is only for diagnostics."""
        rows = rows or slice(None)
        codes = self.codes(module)[rows]
        scales = self.raw_scales(module)[rows]
        if not signed_scales:
            scales = scales.abs().clamp_min(SCALE_EPS)
        signs = unpack_signs(codes).float()
        out = signs.view(codes.shape[0], module.num_groups, module.group_size) * scales.unsqueeze(
            -1
        )
        return out.reshape(codes.shape[0], module.in_features).to(dtype)
