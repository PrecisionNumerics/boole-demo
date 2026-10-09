#!/usr/bin/env python
"""Compare llama.cpp (Q1_0 GGUF) and vLLM + Humming against the dense bf16 reference.

All three engines receive identical token ids, so tokenizer differences cannot hide (or fake)
numerical ones. For each prompt we compare the next-token distribution (top-k log-probs), a
greedy continuation, and -- on prefixes of one passage -- the log-prob of the true next token.

    python scripts/compare_engines.py all --checkpoint CKPT --vllm-dir DIR --gguf FILE \
        --llama-cpp third_party/llama.cpp --tokenizer Qwen/Qwen3-1.7B --out-dir out/compare

Each engine runs in its own process (stage) and writes <out-dir>/<engine>.json; ``report``
reads them. Stages can be run individually, e.g. ``vllm`` after re-exporting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TOPK = 20
GREEDY_TOKENS = 32

PLAIN = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "In 1969, the first humans landed on the Moon. The mission was called",
    "Water boils at a temperature of 100 degrees Celsius at sea level, but on top of a mountain",
    "The three primary colors are red, blue, and",
    "Once upon a time, in a small village by the sea, there lived a",
    "Q: What is 17 multiplied by 23?\nA:",
    "The mitochondria is often described as the",
    "import numpy as np\n\nx = np.arange(10)\nprint(x.",
    "Photosynthesis is the process by which plants convert",
]
CHAT = [
    "Explain what a hash table is in two sentences.",
    "Write a haiku about autumn.",
    "What is the derivative of x^3 + 2x?",
    "Give me three tips for learning a new language.",
    "Translate 'good morning' into Spanish and French.",
    "Why is the sky blue?",
]
PASSAGE = (
    "The city council met on Tuesday evening to discuss the proposed changes to the public "
    "transit system. After several hours of debate, members voted to extend bus service to the "
    "eastern suburbs and to add two new express routes connecting the university campus with the "
    "downtown business district. Supporters argued that the improvements would reduce traffic "
    "congestion and make it easier for students and workers to commute without a car, while "
    "critics worried about the cost to taxpayers."
)


TOKENIZER_PROBES = [
    "Hello, world!",
    "1234567890 and 3.14159, 1,000,000",
    "don't can't I'll we've they're It's",
    "multiple   spaces\tand\ttabs\n\nand blank lines\n",
    "    indented code block\n        deeper",
    "def f(x):\n    return x ** 2  # square\n",
    "https://example.com/path?query=1&b=2#frag",
    "emoji \U0001f600\U0001f680 and accents: caf\u00e9, na\u00efve, \u00fcber",
    "\u4f60\u597d\uff0c\u4e16\u754c\uff01\u6211\u7231\u673a\u5668\u5b66\u4e60\u3002",
    "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"
    " and \u043f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440",
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n",
    "<think>\n\n</think>\n\nAnswer: 42.",
    "UPPER lower MiXeD camelCaseIdentifier snake_case_identifier",
    "x = [1, 2, 3]; y = {'a': 1}; z = (x, y) -> {}",
    "End of text.  ",
]


def build_cases(tokenizer) -> list[dict]:
    cases = [{"name": f"plain{i}", "ids": tokenizer(t).input_ids} for i, t in enumerate(PLAIN)]
    for i, q in enumerate(CHAT):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        cases.append({"name": f"chat{i}", "ids": tokenizer(text).input_ids})
    passage = tokenizer(PASSAGE).input_ids
    for cut in range(8, len(passage) - 1, 8):
        cases.append({"name": f"passage@{cut}", "ids": passage[:cut], "true_next": passage[cut]})
    return cases


def _record(top: dict[int, float], greedy: list[int], true_lp: float | None = None) -> dict:
    return {"top": {str(k): v for k, v in top.items()}, "greedy": greedy, "true_lp": true_lp}


# --------------------------------------------------------------------------- stages


def stage_reference(args) -> dict:
    import torch
    from transformers import AutoTokenizer

    from boole_demo.reference import build_reference_model
    from boole_demo.w1 import W1Checkpoint

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = build_reference_model(W1Checkpoint(args.checkpoint))
    results = {}
    for case in build_cases(tok):
        ids = torch.tensor([case["ids"]], device="cuda")
        with torch.no_grad():
            lp = torch.log_softmax(model(ids).logits[0, -1].float(), dim=-1)
            out = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=GREEDY_TOKENS,
                do_sample=False,
                pad_token_id=0,
            )[0, ids.shape[1] :].tolist()
        vals, idx = lp.topk(TOPK)
        true_lp = lp[case["true_next"]].item() if "true_next" in case else None
        results[case["name"]] = _record(
            dict(zip(idx.tolist(), vals.tolist(), strict=True)), out, true_lp
        )
    return results


def stage_vllm(args) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cases = build_cases(tok)
    llm = LLM(
        model=str(args.vllm_dir),
        max_model_len=2048,
        gpu_memory_utilization=0.4,
        enable_prefix_caching=False,
    )
    params = SamplingParams(temperature=0, max_tokens=GREEDY_TOKENS, logprobs=TOPK)
    outs = llm.generate([{"prompt_token_ids": c["ids"]} for c in cases], params)
    results = {}
    for case, out in zip(cases, outs, strict=True):
        comp = out.outputs[0]
        top = {tid: lp.logprob for tid, lp in comp.logprobs[0].items()}
        results[case["name"]] = _record(top, list(comp.token_ids))
    # True-next log-probs come from a second, prompt_logprobs request over the full passage.
    passage = tok(PASSAGE).input_ids
    po = llm.generate(
        [{"prompt_token_ids": passage}], SamplingParams(max_tokens=1, prompt_logprobs=0)
    )[0].prompt_logprobs
    for case in cases:
        if "true_next" in case:
            pos = len(case["ids"])
            results[case["name"]]["true_lp"] = po[pos][case["true_next"]].logprob
    return results


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def stage_llama(args) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cases = build_cases(tok)
    port = 8089
    server = subprocess.Popen(
        [
            str(args.llama_cpp / "build/bin/llama-server"),
            "-m",
            str(args.gguf),
            "-ngl",
            "99",
            "-c",
            "4096",
            "-np",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(240):
            try:
                if (
                    json.load(urllib.request.urlopen(f"{base}/health", timeout=2)).get("status")
                    == "ok"
                ):
                    break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server did not become healthy")
        # tokenizer parity, including special tokens, digits, whitespace runs, CJK, emoji, code
        mismatches = [
            text
            for text in TOKENIZER_PROBES
            if _post(f"{base}/tokenize", {"content": text, "parse_special": True})["tokens"]
            != tok(text).input_ids
        ]
        results = {"__tokenizer_parity__": [len(TOKENIZER_PROBES), len(mismatches)]}
        for case in cases:
            r = _post(
                f"{base}/completion",
                {
                    "prompt": case["ids"],
                    "n_predict": GREEDY_TOKENS,
                    "temperature": 0,
                    "n_probs": TOPK,
                    "cache_prompt": False,
                },
            )
            probs = r["completion_probabilities"]
            top = {e["id"]: e["logprob"] for e in probs[0]["top_logprobs"]}
            # The server reports only the top-k; the true token is usually, not always, inside it.
            true_lp = top.get(case["true_next"]) if "true_next" in case else None
            generated = [step["id"] for step in probs]  # `tokens` is empty unless requested
            results[case["name"]] = _record(top, generated, true_lp)
        return results
    finally:
        server.terminate()
        server.wait(timeout=30)


# --------------------------------------------------------------------------- report


def _top1(record: dict) -> str:
    return max(record["top"], key=record["top"].get)


def _greedy_match(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):  # a run may stop early at EOS
        if x != y:
            break
        n += 1
    return n


def stage_report(args) -> str:
    ref = json.loads((args.out_dir / "reference.json").read_text())
    engines = {"dense bf16 reference": ref}
    notes = []
    for key, label in (("vllm", "vLLM + Humming"), ("llama", "llama.cpp Q1_0")):
        path = args.out_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            if "__tokenizer_parity__" in data:
                total, bad = data.pop("__tokenizer_parity__")
                notes.append(
                    f"llama.cpp vs HF tokenizer: {total - bad}/{total} probe strings identical "
                    "(special tokens, digits, whitespace, CJK, emoji, code)"
                )
            engines[label] = data

    # NLL only over positions where every engine has the true token's log-prob.
    passage = [n for n in ref if n.startswith("passage@")]
    common = [n for n in passage if all(e[n]["true_lp"] is not None for e in engines.values())]

    rows = [
        "| engine | top-1 = reference | mean abs d(logprob) on ref top-5 | max abs d | "
        "greedy prefix match (mean/32) | identical 32 tokens | passage NLL |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, data in engines.items():
        nll = -sum(data[n]["true_lp"] for n in common) / max(len(common), 1)
        if data is ref:
            rows.append(f"| {label} | - | - | - | - | - | {nll:.4f} |")
            continue
        agree, deltas, matched = 0, [], []
        for name, r in ref.items():
            g = data[name]
            agree += int(_top1(r) == _top1(g))
            for tid, lp in sorted(r["top"].items(), key=lambda kv: -kv[1])[:5]:
                if tid in g["top"]:
                    deltas.append(abs(g["top"][tid] - lp))
            matched.append(_greedy_match(r["greedy"], g["greedy"]))
        rows.append(
            f"| {label} | {agree}/{len(ref)} | {sum(deltas) / len(deltas):.4f} | "
            f"{max(deltas):.4f} | {sum(matched) / len(matched):.1f} | "
            f"{sum(m == GREEDY_TOKENS for m in matched)}/{len(ref)} "
            f"| {nll:.4f} |"
        )
    text = "\n".join(
        rows + ["", f"passage NLL over {len(common)} common positions; {len(ref)} prompts."] + notes
    )
    (args.out_dir / "report.md").write_text(text + "\n")
    return text


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("stage", choices=["all", "reference", "vllm", "llama", "report"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vllm-dir", type=Path)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--llama-cpp", type=Path, default=Path("third_party/llama.cpp"))
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--out-dir", type=Path, default=Path("out/compare"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "all":
        base = [sys.executable, __file__]
        passthrough = [a for a in sys.argv[2:]]
        for stage in ("reference", "vllm", "llama"):
            print(f"== {stage}", flush=True)
            subprocess.run(base + [stage] + passthrough, check=True)
        args.stage = "report"
    if args.stage == "report":
        print(stage_report(args))
        return
    results = {"reference": stage_reference, "vllm": stage_vllm, "llama": stage_llama}[args.stage](
        args
    )
    (args.out_dir / f"{args.stage}.json").write_text(json.dumps(results))
    print(f"wrote {args.out_dir / (args.stage + '.json')}")


if __name__ == "__main__":
    main()
