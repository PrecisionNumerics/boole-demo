"""Quantization method ``w1_humming``: vLLM's Humming backend plus 1-bit vocabulary layers.

Stock vLLM Humming serves ``Linear`` layers only: it always skips ``lm_head`` and has no
embedding method. This subclass hands every linear layer to the stock implementation and adds

* :class:`W1EmbeddingMethod` -- gathers rows of the packed sign codes and applies the
  embedding's own group scales, and
* :class:`W1HeadMethod` -- runs the output projection through the same Humming GEMM, with
  ``tie_weights`` sharing the embedding's *codes* while the head keeps its *own* scales.

Written against the public vLLM 0.31.0 and Humming kernels 0.1.16 packages.
"""

from __future__ import annotations

import torch
import vllm.utils.humming as _hm
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.humming import (
    HummingConfig,
    HummingLayerQuantizationConfig,
    HummingLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter

from . import QUANT_METHOD, ZERO_POINT

_BITS_PER_WORD = 32


class W1HummingConfig(HummingConfig):
    @classmethod
    def get_name(cls) -> str:  # type: ignore[override]
        return QUANT_METHOD

    @classmethod
    def from_config(cls, config: dict) -> W1HummingConfig:
        # The stock Humming machinery is configured by a plain "humming" schema.
        if config.get("dtype") != "uint1" or not config.get("is_fp_zero_point"):
            raise ValueError("w1_humming requires dtype=uint1 with an fp zero point")
        if config.get("zero_point") != ZERO_POINT:
            raise ValueError(f"w1_humming requires the constant zero_point {ZERO_POINT}")
        return cls(full_config={**config, "quant_method": "humming"})

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        return None

    def _vocab_config(self) -> HummingLayerQuantizationConfig:
        return HummingLayerQuantizationConfig(
            weight_schema=_hm.BaseWeightSchema.from_config(self.full_config)
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, ParallelLMHead):  # before its base class
            return W1HeadMethod(W1LinearMethod(self._vocab_config()))
        if isinstance(layer, VocabParallelEmbedding):
            return W1EmbeddingMethod(group_size=self.full_config["group_size"])
        method = super().get_quant_method(layer, prefix)
        if type(method) is HummingLinearMethod:
            return W1LinearMethod(method.quant_config)
        return method


class W1LinearMethod(HummingLinearMethod):
    """Stock Humming linear whose fp zero point is the constant 0.5 (bipolar +/-1 signs).

    The tensor is not stored in the checkpoint. It is filled when the parameter is created, so a
    checkpoint that does carry ``zero_point`` still overrides it when loaded.
    """

    def create_weights(self, layer: torch.nn.Module, *args, **kwargs) -> None:
        super().create_weights(layer, *args, **kwargs)
        layer.zero_point.data.fill_(ZERO_POINT)


class W1EmbeddingMethod(QuantizeMethodBase):
    """Input embedding stored as packed signs + per-group scale (zero point is the constant 0.5)."""

    def __init__(self, *, group_size: int):
        self.group_size = group_size

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        hidden = input_size_per_partition
        if hidden % self.group_size or hidden % _BITS_PER_WORD:
            raise ValueError(f"hidden size {hidden} is not a multiple of the group size")
        loader = extra_weight_attrs["weight_loader"]
        rows = sum(output_partition_sizes)
        groups = hidden // self.group_size
        layer.register_parameter(
            "weight",
            PackedvLLMParameter(
                data=torch.empty(rows, hidden // _BITS_PER_WORD, dtype=torch.int32),
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=_BITS_PER_WORD,
                weight_loader=loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            GroupQuantScaleParameter(
                data=torch.empty(rows, groups, dtype=params_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=loader,
            ),
        )

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        ids = input_.reshape(-1)
        words = layer.weight[ids]  # [T, K/32] int32
        # int32 -> bytes (little-endian) restores the LSB-first uint8 stream.
        codes = words.view(torch.uint8)  # [T, K/8]
        shifts = torch.arange(8, device=codes.device, dtype=torch.uint8)
        bits = ((codes.unsqueeze(-1) >> shifts) & 1).reshape(ids.numel(), -1, self.group_size)
        scale = layer.weight_scale[ids].float().unsqueeze(-1)
        out = (bits.float() - ZERO_POINT) * scale
        return out.reshape(*input_.shape, -1).to(layer.weight_scale.dtype)

    def apply(self, layer, x, bias=None):
        raise NotImplementedError("the embedding is only used for lookups")


class W1HeadMethod(QuantizeMethodBase):
    """Output head: the stock Humming GEMM over shared sign codes and the head's own scales."""

    def __init__(self, linear: W1LinearMethod):
        self.linear = linear

    def create_weights(self, layer, *args, **kwargs) -> None:
        self.linear.create_weights(layer, *args, **kwargs)

    def tie_weights(self, layer: torch.nn.Module, embed_tokens: torch.nn.Module):
        # Share the sign codes only. weight_scale stays the head's own tensor and is loaded from
        # `lm_head.weight_scale`; the embedding keeps its unprocessed codes.
        layer.weight = embed_tokens.weight
        return layer

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.linear.process_weights_after_loading(layer)

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.linear.apply(layer, x, bias)
        # Humming pads N to a kernel tile (151936 -> 152064); vLLM expects exactly the layer's
        # (vocab-padded) local width, which is what a tensor-parallel gather also assumes.
        return logits[..., : layer.num_embeddings_per_partition]
