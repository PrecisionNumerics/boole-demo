"""Reader for the packed 1-bit ("W1") checkpoint.

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


class W1FormatError(ValueError):
    """The file does not follow the packed 1-bit layout described above."""


@dataclass(frozen=True)
class W1Module:
    name: str
    out_features: int
    in_features: int
    group_size: int
    code_key: str
    scale_key: str


class W1Checkpoint:
    """A validated view of the packed safetensors file (tensors are read lazily)."""

    def __init__(self, path: str | Path):
        self._file = safe_open(Path(path), framework="pt", device="cpu")
        self.metadata: dict[str, str] = dict(self._file.metadata() or {})
        self._keys = set(self._file.keys())
        self.modules: tuple[W1Module, ...] = self._discover()
        claimed = {m.code_key for m in self.modules} | {m.scale_key for m in self.modules}
        self.dense_keys: tuple[str, ...] = tuple(sorted(self._keys - claimed))

    def __enter__(self) -> W1Checkpoint:
        return self

    def __exit__(self, *exc) -> None:
        self._file = None  # safe_open handles are released with the object

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
            if code_key != weight_key and name != "lm_head":
                raise W1FormatError(f"{name!r}: only the output head may alias another module")
            modules.append(
                W1Module(
                    name=name,
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

    def tensor(self, key: str) -> torch.Tensor:
        return self._file.get_tensor(key)

    def codes(self, module: W1Module) -> torch.Tensor:
        """Packed uint8 signs ``[N, K/8]``; shared by the embedding and the head."""
        return self.tensor(module.code_key).contiguous()

    def magnitudes(self, module: W1Module) -> torch.Tensor:
        """Effective per-group magnitudes ``max(|scale|, SCALE_EPS)`` as float32 ``[N, G]``."""
        return self.tensor(module.scale_key).float().abs().clamp_min(SCALE_EPS)
