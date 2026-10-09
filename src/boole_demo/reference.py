"""Dense bf16 Qwen3 rebuilt from the packed weights: the numerical reference.

Every engine in this repository (llama.cpp, vLLM + Humming) is compared against this model,
which uses the plain transformers implementation and no quantized kernels.
"""

from __future__ import annotations

import torch

from .qwen3 import hf_config
from .w1 import W1Checkpoint


def build_reference_model(
    ckpt: W1Checkpoint,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    signed_scales: bool = False,
    base_config: dict | None = None,
):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = hf_config(ckpt, base_config)
    # The head has its own scales, so it is a separate dense matrix here.
    config["tie_word_embeddings"] = False
    config.pop("torch_dtype", None)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(Qwen3Config(**config))
    state = {}
    for module in ckpt.modules:
        state[f"{module.name}.weight"] = ckpt.dense(module, dtype, signed_scales=signed_scales).to(
            device
        )
    for key in ckpt.dense_keys:
        state[key] = ckpt.tensor(key).to(device=device, dtype=dtype)
    model.load_state_dict(state, strict=True, assign=True)
    # Rotary tables are non-persistent buffers and were left on the meta device.
    model.model.rotary_emb = type(model.model.rotary_emb)(config=model.config).to(device)
    return model.eval()
