"""vLLM general-plugin entry point (``vllm.general_plugins``: ``boole_w1``).

Kept free of heavy imports: vLLM runs this in every engine process, and the quantization
classes are only imported when the method is first registered.
"""

from __future__ import annotations

_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        register_quantization_config,
    )

    from .vllm_export import QUANT_METHOD

    if QUANT_METHOD not in QUANTIZATION_METHODS:
        from .vllm_quant import W1HummingConfig

        register_quantization_config(QUANT_METHOD)(W1HummingConfig)
    _registered = True
