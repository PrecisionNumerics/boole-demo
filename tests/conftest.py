import os
from pathlib import Path

import pytest

from .synthetic import TINY_CONFIG, make_synthetic_checkpoint


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("synthetic") / "tiny-packed.safetensors"
    make_synthetic_checkpoint(path)
    return path


@pytest.fixture(scope="session")
def tiny_config() -> dict:
    return dict(TINY_CONFIG)


@pytest.fixture(scope="session")
def real_checkpoint() -> Path:
    path = os.environ.get("BOOLE_CHECKPOINT")
    if not path or not Path(path).is_file():
        pytest.skip("set BOOLE_CHECKPOINT to the opd1200 packed safetensors file")
    return Path(path)


def pytest_collection_modifyitems(config, items):
    import torch

    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="requires a CUDA device")
    for item in items:
        if "gpu" in item.keywords or "vllm" in item.keywords:
            item.add_marker(skip)
