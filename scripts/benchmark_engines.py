#!/usr/bin/env python
"""Benchmark warm, fixed-length local serving for llama.cpp and vLLM + Humming.

Both engines receive identical prompt token IDs. Rates include prefill, generation,
and local HTTP overhead, but exclude model loading, compilation, and warmup.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

PASSAGE = (
    "The city council met on Tuesday evening to discuss improvements to the public transit "
    "system. Members proposed extending bus service to the eastern suburbs and adding two "
    "express routes between the university and downtown. Explain the benefits, costs, and "
    "tradeoffs in detail, including how these changes could affect students and workers. "
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompts(args) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.vllm_dir)
    return [
        tokenizer(
            f"Request {i}: " + PASSAGE * args.input_tokens, add_special_tokens=False
        ).input_ids[: args.input_tokens]
        for i in range(max(args.batches))
    ]


def request(engine, base, ids, output_tokens, barrier) -> dict:
    if engine == "llama":
        endpoint = "/completion"
        payload = {
            "prompt": ids,
            "n_predict": output_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "cache_prompt": False,
            "return_tokens": True,
            "stream": True,
            "samplers": ["temperature"],
        }
    else:
        endpoint = "/v1/completions"
        payload = {
            "model": "boole-benchmark",
            "prompt": ids,
            "max_tokens": output_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    req = urllib.request.Request(
        base + endpoint, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    barrier.wait(timeout=30)
    started = time.perf_counter()
    first = last = None
    first_count = 0
    streamed_tokens = 0
    timings = usage = None
    declared_input = declared_output = None
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                if engine == "llama":
                    # The terminal response repeats all token IDs; count delta chunks only.
                    count = 0 if chunk.get("stop") else len(chunk.get("tokens", []))
                    if chunk.get("stop"):
                        timings = chunk["timings"]
                        declared_input = chunk["tokens_evaluated"]
                        declared_output = chunk["tokens_predicted"]
                else:
                    count = sum(len(c.get("token_ids") or []) for c in chunk.get("choices", []))
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                        declared_input = usage["prompt_tokens"]
                        declared_output = usage["completion_tokens"]
                if count:
                    now = time.perf_counter()
                    if first is None:
                        first, first_count = now, count
                    last = now
                    streamed_tokens += count
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error
    finished = time.perf_counter()
    if declared_input != len(ids) or declared_output != output_tokens:
        raise RuntimeError(f"Unexpected token counts: {declared_input=}, {declared_output=}")
    if first is None or last is None or streamed_tokens != output_tokens:
        raise RuntimeError(f"Unexpected streamed token count: {streamed_tokens}")
    if timings is not None and timings["prompt_n"] != len(ids):
        raise RuntimeError(f"Prompt cache reused: {timings}")
    if usage is not None and (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0):
        raise RuntimeError(f"Prompt cache reused: {usage}")
    return {
        "input_tokens": declared_input,
        "output_tokens": declared_output,
        "elapsed_s": finished - started,
        "ttft_s": first - started,
        "decode_tokens_per_s": (output_tokens - first_count) / (last - first),
        "timings": timings,
        "usage": usage,
    }


def batch(engine, base, ids, output_tokens) -> dict:
    barrier = threading.Barrier(len(ids) + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ids)) as executor:
        futures = [executor.submit(request, engine, base, p, output_tokens, barrier) for p in ids]
        started = time.perf_counter()
        barrier.wait(timeout=30)
        results = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
    return {
        "elapsed_s": elapsed,
        "output_tokens_per_s": sum(r["output_tokens"] for r in results) / elapsed,
        "median_ttft_ms": statistics.median(r["ttft_s"] for r in results) * 1000,
        "median_decode_tokens_per_s": statistics.median(r["decode_tokens_per_s"] for r in results),
        "requests": results,
    }


def command(args, engine) -> list[str]:
    if engine == "llama":
        return [
            str(args.llama_cpp / "build/bin/llama-server"),
            "-m",
            str(args.gguf),
            "-ngl",
            "99",
            "-fa",
            "on",
            "-t",
            "8",
            "-np",
            str(max(args.batches)),
            "-c",
            str(1024 * max(args.batches)),
            "-b",
            "2048",
            "-ub",
            "512",
            "--cache-ram",
            "0",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(args.vllm_dir),
        "--served-model-name",
        "boole-benchmark",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        str(max(args.batches)),
        "--max-num-batched-tokens",
        "2048",
        "--gpu-memory-utilization",
        "0.6",
        "--no-enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--dtype",
        "bfloat16",
        "--seed",
        "0",
        "--disable-uvicorn-access-log",
    ]


def run(args, engine, ids) -> dict:
    cmd = command(args, engine)
    base = f"http://127.0.0.1:{args.port}"
    log_path = args.out_dir / f"{engine}-server.log"
    print(f"Starting {engine}; log: {log_path}", flush=True)
    with log_path.open("w") as log:
        server = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"{engine} exited; see {log_path}")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError(f"{engine} did not become ready; see {log_path}")
            measured = []
            for size in args.batches:
                for _ in range(args.warmups):
                    batch(engine, base, ids[:size], args.output_tokens)
                samples = []
                for repetition in range(args.repetitions):
                    sample = batch(engine, base, ids[:size], args.output_tokens)
                    samples.append(sample)
                    print(
                        f"{engine} batch={size} repeat={repetition + 1}: "
                        f"{sample['output_tokens_per_s']:.1f} output tok/s, "
                        f"TTFT={sample['median_ttft_ms']:.1f} ms",
                        flush=True,
                    )
                rates = [s["output_tokens_per_s"] for s in samples]
                measured.append(
                    {
                        "batch_size": size,
                        "output_tokens_per_s_mean": statistics.mean(rates),
                        "output_tokens_per_s_stdev": statistics.stdev(rates)
                        if len(rates) > 1
                        else 0,
                        "median_ttft_ms": statistics.median(s["median_ttft_ms"] for s in samples),
                        "median_decode_tokens_per_s": statistics.median(
                            s["median_decode_tokens_per_s"] for s in samples
                        ),
                        "samples": samples,
                    }
                )
            return {"engine": engine, "command": cmd, "results": measured}
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
            except ProcessLookupError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=["all", "llama", "vllm"])
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--vllm-dir", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, default=Path("third_party/llama.cpp"))
    parser.add_argument("--out-dir", type=Path, default=Path("out/benchmark"))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    if min(*args.batches, args.input_tokens, args.output_tokens, args.repetitions) < 1:
        parser.error("batch sizes, lengths, and repetitions must be positive")
    if args.input_tokens + args.output_tokens > 1024 or args.output_tokens < 2:
        parser.error("prompt + output must fit 1024 tokens; output must be at least 2 tokens")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = prompts(args)
    metadata = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "versions": {n: version(n) for n in ("torch", "vllm", "humming-kernels")},
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,power.limit",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "llama_commit": subprocess.check_output(
            [
                "git",
                "-C",
                str(args.llama_cpp),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        "gguf_sha256": sha256(args.gguf),
        "vllm_weights_sha256": sha256(args.vllm_dir / "model.safetensors"),
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "warmups_per_batch": args.warmups,
        "repetitions": args.repetitions,
        "prompt_token_ids": ids,
        "metric": (
            "generated output tokens / concurrent-batch wall time, including prefill and HTTP"
        ),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for engine in ["llama", "vllm"] if args.engine == "all" else [args.engine]:
        results = run(args, engine, ids)
        (args.out_dir / f"{engine}.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
