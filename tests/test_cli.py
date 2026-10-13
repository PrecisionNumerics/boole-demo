import json

import pytest

from boole_demo import cli

from .synthetic import TINY_CONFIG, write_checkpoint


@pytest.mark.parametrize("model_args", [[], ["precision-numerics/Boole-1.7B"]])
def test_cli_downloads_and_exports_safetensors_repo(model_args, tmp_path, monkeypatch):
    model = tmp_path / "downloaded"
    model.mkdir()
    write_checkpoint(model / "Boole-tiny.safetensors")
    (model / "config.json").write_text(json.dumps(TINY_CONFIG))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (model / name).write_text("{}")
    downloads = []

    def download(repo_id, *, allow_patterns):
        downloads.append(repo_id)
        assert "*.safetensors" in allow_patterns
        assert {"config.json", "tokenizer.json", "tokenizer_config.json"} <= set(allow_patterns)
        assert not any("gguf" in pattern for pattern in allow_patterns)
        return str(model)

    monkeypatch.setattr(cli, "snapshot_download", download)
    out = tmp_path / "out"
    cli.main(["vllm", *model_args, "--out", str(out)])

    assert downloads == ["precision-numerics/Boole-1.7B"]
    assert (out / "model.safetensors").is_file()
    assert (out / "config.json").is_file()
    assert (out / "tokenizer.json").is_file()


def test_cli_rejects_gguf_repo_before_downloading(tmp_path, monkeypatch, capsys):
    def download(*args, **kwargs):
        pytest.fail("The GGUF repository must not be downloaded for a vLLM export")

    monkeypatch.setattr(cli, "snapshot_download", download)
    with pytest.raises(SystemExit) as error:
        cli.main(["vllm", "precision-numerics/Boole-1.7B-GGUF", "--out", str(tmp_path / "out")])

    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "precision-numerics/Boole-1.7B" in message
    assert "safetensors" in message
    assert "llama.cpp" in message
