"""Copy the model repository's tokenizer and generation settings into an export."""

from __future__ import annotations

import shutil
from pathlib import Path

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "chat_template.jinja",
)


def copy_tokenizer_files(dest: Path, source: Path) -> None:
    for name in TOKENIZER_FILES:
        path = source / name
        if path.is_file():
            shutil.copyfile(path, dest / name)
        elif name in ("tokenizer.json", "tokenizer_config.json"):
            raise FileNotFoundError(f"{path} is required")
