"""Tokenizer and generation files that accompany an exported model."""

from __future__ import annotations

import shutil
from pathlib import Path

DEFAULT_TOKENIZER_REPO = "Qwen/Qwen3-1.7B"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
)


def copy_tokenizer_files(dest: Path, source: str | Path = DEFAULT_TOKENIZER_REPO) -> None:
    """Copy the tokenizer/generation files from a local directory or a Hugging Face repo id.

    The checkpoint itself does not carry a tokenizer; the student shares Qwen3's vocabulary.
    """
    dest.mkdir(parents=True, exist_ok=True)
    local = Path(source)
    if local.is_dir():
        for name in TOKENIZER_FILES:
            if (local / name).is_file():
                shutil.copyfile(local / name, dest / name)
            elif name in ("tokenizer.json", "tokenizer_config.json"):
                raise FileNotFoundError(f"{local / name} is required")
        return
    from huggingface_hub import hf_hub_download

    for name in TOKENIZER_FILES:
        # copyfile, not copy2: hub cache blobs are read-only and must not poison re-runs
        shutil.copyfile(hf_hub_download(str(source), name), dest / name)
