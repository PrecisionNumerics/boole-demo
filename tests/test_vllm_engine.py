"""vLLM + Humming on a tiny synthetic checkpoint, against the dense bf16 reference.

The synthetic vocabulary (600) is not a multiple of vLLM's vocab padding (64) or Humming's
N tile (256), and the head's scales differ from the embedding's -- the same traps as the real
model. Tolerances are justified by the negative controls below: wiring mistakes that matter
(signed scales, a head reusing the embedding's scales) move the log-probs by far more.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from boole_demo.reference import build_reference_model
from boole_demo.vllm_export import export_vllm_dir
from boole_demo.w1 import W1Checkpoint

pytestmark = [pytest.mark.vllm, pytest.mark.gpu]

TOLERANCE = 0.06  # max |d logprob|; measured ~0.025 (bf16 end to end), controls are > 1.4


def _prompts(vocab: int) -> list[list[int]]:
    gen = torch.Generator().manual_seed(7)
    prompts = [torch.randint(0, vocab, (24,), generator=gen).tolist() for _ in range(5)]
    prompts.append(list(range(vocab - 12, vocab)))  # ids in the padded tail region
    return prompts


def _reference_logprobs(model, prompt):
    ids = torch.tensor([prompt], device="cuda")
    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[0].float(), dim=-1)
    teacher = [lp[i - 1, prompt[i]].item() for i in range(1, len(prompt))]
    return teacher, lp[-1]


def _max_gap(model, prompts, engine) -> float:
    gap = 0.0
    for prompt, got in zip(prompts, engine["prompts"], strict=True):
        teacher, last = _reference_logprobs(model, prompt)
        gap = max(gap, max(abs(a - b) for a, b in zip(teacher, got["teacher"][1:], strict=True)))
        gap = max(gap, max(abs(last[int(t)].item() - v) for t, v in got["next"].items()))
    return gap


@pytest.fixture(scope="module")
def engine_run(tiny_checkpoint, tiny_config, tmp_path_factory):
    work = tmp_path_factory.mktemp("engine")
    ckpt = W1Checkpoint(tiny_checkpoint)
    model_dir = export_vllm_dir(ckpt, work / "model", tokenizer=None, base_config=tiny_config)
    prompts = _prompts(tiny_config["vocab_size"])
    (work / "prompts.json").write_text(json.dumps(prompts))
    runs = {}
    for mode, eager in (("compiled", "0"), ("eager", "1")):
        out = work / f"{mode}.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("_vllm_worker.py")),
                str(model_dir),
                str(work / "prompts.json"),
                str(out),
                eager,
            ],
            check=True,
        )
        runs[mode] = json.loads(out.read_text())
    return ckpt, prompts, runs


@pytest.mark.parametrize("mode", ["compiled", "eager"])
def test_quant_methods_are_wired(engine_run, mode):
    _, _, runs = engine_run
    assert runs[mode]["methods"] == {
        "embed": "W1EmbeddingMethod",
        "head": "W1HeadMethod",
        "qkv": "W1LinearMethod",  # stock HummingLinearMethod + constant zero point
    }
    assert runs[mode]["head_scale_differs"]


@pytest.mark.parametrize("mode", ["compiled", "eager"])
def test_matches_dense_reference(engine_run, tiny_config, mode):
    ckpt, prompts, runs = engine_run
    model = build_reference_model(ckpt, base_config=tiny_config)
    gap = _max_gap(model, prompts, runs[mode])
    print(f"max |d logprob| vs dense reference ({mode}): {gap:.4f} (tolerance {TOLERANCE})")
    assert gap < TOLERANCE


def test_negative_controls_are_far_outside_tolerance(engine_run, tiny_config):
    ckpt, prompts, runs = engine_run
    engine = runs["eager"]
    signed = build_reference_model(ckpt, base_config=tiny_config, signed_scales=True)
    gap = _max_gap(signed, prompts, engine)
    print(f"control: signed scales -> max |d logprob| {gap:.4f}")
    assert gap > 5 * TOLERANCE

    shared_scales = build_reference_model(ckpt, base_config=tiny_config)
    embed = ckpt.module("model.embed_tokens")
    shared_scales.lm_head.weight.data = ckpt.dense(embed, torch.bfloat16).cuda()
    gap = _max_gap(shared_scales, prompts, engine)
    print(f"control: head reuses embedding scales -> max |d logprob| {gap:.4f}")
    assert gap > 5 * TOLERANCE
