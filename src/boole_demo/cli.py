"""``boole-prep``: prepare the packed 1-bit checkpoint for llama.cpp and vLLM + Humming."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .model_dir import DEFAULT_TOKENIZER_REPO
from .qwen3 import infer_dims, load_base_config
from .w1 import W1Checkpoint


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("checkpoint", type=Path, help="packed safetensors file")
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER_REPO,
        help="directory or Hugging Face repo id providing tokenizer/generation files "
        f"(default: {DEFAULT_TOKENIZER_REPO})",
    )
    parser.add_argument(
        "--base-config", type=Path, help="config.json to use instead of Qwen3-1.7B's"
    )


def cmd_inspect(args: argparse.Namespace) -> None:
    ckpt = W1Checkpoint(args.checkpoint)
    kinds = Counter(m.kind for m in ckpt.modules)
    print(f"{args.checkpoint}: {dict(kinds)} dense tensors: {len(ckpt.dense_keys)}")
    print(f"group size {ckpt.group_size}; metadata {ckpt.metadata}")
    print(json.dumps(infer_dims(ckpt), indent=2))
    head, embed = ckpt.module("lm_head"), ckpt.module("model.embed_tokens")
    shared = head.code_key == embed.code_key
    same = bool((ckpt.raw_scales(head) == ckpt.raw_scales(embed)).all())
    print(
        f"head shares embedding codes: {shared}; head scales identical to embedding scales: {same}"
    )
    raw = [ckpt.raw_scales(m) for m in ckpt.modules]
    total = sum(t.numel() for t in raw)
    print(
        f"scales: {total} total, {sum(int((t < 0).sum()) for t in raw)} negative, "
        f"{sum(int((t.abs() < 1e-5).sum()) for t in raw)} below the 1e-5 floor"
    )


def cmd_vllm(args: argparse.Namespace) -> None:
    from .vllm_export import export_vllm_dir

    out = export_vllm_dir(
        W1Checkpoint(args.checkpoint),
        args.out,
        tokenizer=args.tokenizer,
        base_config=load_base_config(args.base_config),
    )
    print(f"wrote vLLM + Humming model directory: {out}")


def cmd_gguf(args: argparse.Namespace) -> None:
    from .gguf_export import export_gguf, verify_gguf

    ckpt = W1Checkpoint(args.checkpoint)
    out = export_gguf(
        ckpt,
        args.out,
        args.llama_cpp,
        tokenizer=args.tokenizer,
        base_config=load_base_config(args.base_config),
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if not args.no_verify:
        print("verified against source:", verify_gguf(ckpt, out))


def cmd_all(args: argparse.Namespace) -> None:
    stem = args.checkpoint.stem
    args.out = args.out_dir / f"{stem}-humming"
    cmd_vllm(args)
    args.out = args.out_dir / f"{stem}-Q1_0.gguf"
    cmd_gguf(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="boole-prep", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="validate the checkpoint and print its inventory")
    p.add_argument("checkpoint", type=Path)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("vllm", help="write a vLLM + Humming model directory")
    _common(p)
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.set_defaults(func=cmd_vllm)

    p = sub.add_parser("gguf", help="write a llama.cpp Q1_0 GGUF")
    _common(p)
    p.add_argument("--out", type=Path, required=True, help="output .gguf file")
    p.add_argument(
        "--llama-cpp",
        type=Path,
        default=Path("third_party/llama.cpp"),
        help="llama.cpp checkout providing convert_hf_to_gguf.py (see scripts/setup_llama_cpp.sh)",
    )
    p.add_argument("--no-verify", action="store_true", help="skip the tensor-by-tensor check")
    p.set_defaults(func=cmd_gguf)

    p = sub.add_parser("all", help="write both outputs under --out-dir")
    _common(p)
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    p.add_argument("--llama-cpp", type=Path, default=Path("third_party/llama.cpp"))
    p.add_argument("--no-verify", action="store_true")
    p.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
