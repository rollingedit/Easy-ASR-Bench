import json
from pathlib import Path

from app.model_scanner import scan_models


def test_hf_whisper_safetensors_detected(tmp_path: Path):
    model = tmp_path / "whisper"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "whisper"}), encoding="utf-8")
    (model / "model.safetensors").write_text("", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    runnable, unsupported = scan_models(tmp_path)
    assert any(candidate.adapter_name == "hf_whisper_asr" for candidate in runnable)


def test_whisper_cpp_detected(tmp_path: Path):
    (tmp_path / "ggml-base.bin").write_text("", encoding="utf-8")
    runnable, unsupported = scan_models(tmp_path)
    assert any(candidate.adapter_name == "whisper_cpp" for candidate in runnable)


def test_standalone_safetensors_explained(tmp_path: Path):
    (tmp_path / "random.safetensors").write_text("", encoding="utf-8")
    runnable, unsupported = scan_models(tmp_path)
    assert not runnable
    assert any("complete Hugging Face" in candidate.help_text or candidate.missing_files for candidate in unsupported)
