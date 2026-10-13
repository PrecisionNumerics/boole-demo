"""Export a packed safetensors model for vLLM + Humming."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from .model_dir import TOKENIZER_FILES
from .vllm_export import export_vllm_dir
from .w1 import W1Checkpoint

SAFETENSORS_REPO = "precision-numerics/Boole-1.7B"
GGUF_REPO = "precision-numerics/Boole-1.7B-GGUF"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="boole-prep", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("vllm", help="download and export a model for vLLM + Humming")
    export.add_argument(
        "model",
        nargs="?",
        default=SAFETENSORS_REPO,
        help=f"packed safetensors repo ID or downloaded directory (default: {SAFETENSORS_REPO})",
    )
    export.add_argument("--out", type=Path, required=True, help="output model directory")
    args = parser.parse_args(argv)

    source = Path(args.model)
    if not source.is_dir():
        if args.model == GGUF_REPO:
            parser.error(
                f"vLLM export requires packed safetensors weights; use {SAFETENSORS_REPO}. "
                "Run the GGUF release with llama.cpp."
            )
        source = Path(
            snapshot_download(
                args.model, allow_patterns=["*.safetensors", "config.json", *TOKENIZER_FILES]
            )
        )
    checkpoints = sorted(source.glob("*.safetensors"))
    if len(checkpoints) != 1:
        parser.error(f"expected one packed .safetensors file in {source}, found {len(checkpoints)}")

    config_path = source / "config.json"
    tokenizer = source
    has_tokenizer = all((source / name).is_file() for name in TOKENIZER_FILES[:2])
    if not config_path.is_file() or not has_tokenizer:
        base = Path(
            snapshot_download("Qwen/Qwen3-1.7B", allow_patterns=["config.json", *TOKENIZER_FILES])
        )
        if not config_path.is_file():
            config_path = base / "config.json"
        if not has_tokenizer:
            tokenizer = base
    config = json.loads(config_path.read_text())
    with W1Checkpoint(checkpoints[0]) as checkpoint:
        out = export_vllm_dir(checkpoint, args.out, tokenizer=tokenizer, base_config=config)
    print(f"Ready for vLLM + Humming: {out}")


if __name__ == "__main__":
    main()
