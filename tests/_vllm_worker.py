"""Runs vLLM in its own process (engine tests must not share a CUDA context with pytest).

usage: _vllm_worker.py MODEL_DIR PROMPTS.json OUT.json EAGER(0|1)
"""

import json
import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams  # noqa: E402

model_dir, prompts_path, out_path, eager = sys.argv[1:5]
prompts = json.load(open(prompts_path))
llm = LLM(
    model=model_dir,
    skip_tokenizer_init=True,
    max_model_len=512,
    gpu_memory_utilization=0.3,
    enforce_eager=eager == "1",
    enable_prefix_caching=False,
)
outs = llm.generate(
    [{"prompt_token_ids": p} for p in prompts],
    SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0, logprobs=20),
)
worker = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker
model = worker.model_runner.get_model()
embed, head = model.model.embed_tokens, model.lm_head
qkv = model.model.layers[0].self_attn.qkv_proj
result = {
    "methods": {
        "embed": type(embed.quant_method).__name__,
        "head": type(head.quant_method).__name__,
        "qkv": type(qkv.quant_method).__name__,
    },
    "head_scale_differs": bool(
        embed.weight_scale.shape != head.weight_scale.shape
        or not (embed.weight_scale == head.weight_scale).all().item()
    ),
    "prompts": [],
}
for prompt, out in zip(prompts, outs, strict=True):
    teacher = [None] + [out.prompt_logprobs[i][prompt[i]].logprob for i in range(1, len(prompt))]
    result["prompts"].append(
        {
            "teacher": teacher,
            "next": {str(t): lp.logprob for t, lp in out.outputs[0].logprobs[0].items()},
        }
    )
json.dump(result, open(out_path, "w"))
