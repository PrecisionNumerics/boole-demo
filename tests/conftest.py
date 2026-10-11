from pathlib import Path

import pytest

from .synthetic import TINY_CONFIG, write_checkpoint


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> Path:
    return write_checkpoint(tmp_path_factory.mktemp("synthetic") / "tiny-packed.safetensors")


@pytest.fixture(scope="session")
def tiny_config() -> dict:
    return dict(TINY_CONFIG)
