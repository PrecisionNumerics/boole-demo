# Inference throughput

Measured on 2026-10-09 using the existing local `opd1200` exports of the 1-bit Qwen3-1.7B
checkpoint. The current model repository is [pn-admin/abc](https://huggingface.co/pn-admin/abc)
(temporary name). This run did not re-download the model repository; artifact hashes are recorded
in [benchmarks.json](benchmarks.json).

## Matched local serving benchmark

Each request receives 512 input tokens and generates exactly 256 output tokens. Both engines
receive identical token IDs, use greedy sampling, ignore EOS, and disable prompt/prefix cache
reuse. Requests in each batch start together. There is one unmeasured warmup batch per batch size,
followed by three measured repetitions. The script checks every request's input/output token
counts and verifies that prompt tokens were processed without cache reuse.

**Output tokens/sec** means total generated tokens divided by wall time to complete the entire
concurrent batch. It includes prompt processing, generation, scheduling, and local streaming HTTP
overhead; it excludes model loading, compilation, and warmup. These are aggregate rates across
all concurrent requests. The reported spread is the sample standard deviation across the three
repetitions.

| Concurrent requests | llama.cpp Q1_0 output tokens/sec | vLLM + Humming output tokens/sec | vLLM / llama.cpp |
|---|---:|---:|---:|
| 1 | 727.7 ± 1.3 | 633.4 ± 0.4 | 0.87× |
| 8 | 1,800.2 ± 9.0 | 3,445.2 ± 5.1 | 1.91× |
| 32 | 2,129.8 ± 4.4 | 7,489.2 ± 10.1 | 3.52× |

llama.cpp delivered about 15% more output tokens/sec for a single request. vLLM + Humming
delivered higher aggregate throughput at the two tested concurrent batch sizes.

Time to first token, measured at the client (median of the three per-batch medians):

| Concurrent requests | llama.cpp | vLLM + Humming |
|---|---:|---:|
| 1 | 15.5 ms | 11.9 ms |
| 8 | 91.7 ms | 55.6 ms |
| 32 | 287.9 ms | 154.1 ms |

For one request, the client-observed rate between the first and last output token was 759.1
tokens/sec for llama.cpp and 650.3 tokens/sec for vLLM + Humming, excluding time to first token.

## Environment and settings

- One NVIDIA GeForce RTX 5090, 32 GB VRAM, driver 580.95.05, power limit 500 W; CPU is an
  AMD Ryzen 7 7800X3D. The GPU was idle before the run, and engines ran sequentially.
- llama.cpp commit `b42b7e6d3059503dd4bed6ca5dcf342bb6379f67`, CUDA build, `-ngl 99`, Flash
  Attention on, 8 CPU threads, 32 server slots, total context 32768 (1024 per slot), batch 2048,
  microbatch 512, default f16 KV cache, prompt cache disabled.
- vLLM 0.31.0, Humming kernels 0.1.16, torch 2.13.0 with CUDA 13.0; bf16 weights/activations
  and default bf16 KV cache, context 1024 per sequence, maximum 32 sequences, token batch budget
  2048, GPU memory utilization 0.6, prefix caching disabled, compiled execution and CUDA graphs.
  The engine startup log confirms `quantization=w1_humming`.
- Both servers bind to localhost. The client requests streamed token IDs from both servers and
  checks their counts against the final usage totals.

These are measurements of the full inference stacks: llama.cpp uses W1A8 matrix multiplication
for Q1_0, while Humming uses W1A16. They do not isolate the quantized GEMM kernels. Only the
512-input/256-output workload and concurrency levels above were measured; maximum throughput,
long-context performance, and production traffic behavior remain unmeasured.

## Native llama.cpp microbenchmark

`llama-bench`, five repetitions after its built-in warmup, on the same GGUF and GPU:

| Test | Tokens/sec (mean ± standard deviation) |
|---|---:|
| Prompt processing, 512 tokens | 34,321.7 ± 3,936.3 |
| Decode, 256 tokens, initially empty context | 964.6 ± 2.2 |

This benchmark bypasses HTTP and serving overhead. Its decode test starts with an empty context,
so it measures a different workload from the server comparison above. The first prompt-processing
sample was slower than the remaining four; all five samples are retained in the evidence file.

## Reproduce

From the repository root, using the installed `.venv` and existing exports:

```bash
.venv/bin/python scripts/benchmark_engines.py all \
    --gguf out/qwen3-1.7b-opd1200-Q1_0.gguf \
    --vllm-dir out/qwen3-1.7b-opd1200-humming \
    --batches 1 8 32 --input-tokens 512 --output-tokens 256 \
    --warmups 1 --repetitions 3 --out-dir out/benchmark

third_party/llama.cpp/build/bin/llama-bench \
    -m out/qwen3-1.7b-opd1200-Q1_0.gguf \
    -ngl 99 -p 512 -n 256 -pg 512,256 -r 5 -fa on -t 8 -o json \
    > out/benchmark/llama-bench.json 2> out/benchmark/llama-bench.log
```

The serving script starts and stops each engine, including cleanup on failure. Raw per-request
results, prompt token IDs, artifact hashes, engine versions, exact commands, and server logs go
to `out/benchmark/`. [benchmarks.json](benchmarks.json) preserves the settings and individual
batch measurements used for the published tables, plus the native benchmark samples.
