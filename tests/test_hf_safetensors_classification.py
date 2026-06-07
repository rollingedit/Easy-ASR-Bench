import json
from pathlib import Path

from app.adapters.hf_transformers_asr import HFTransformersASRAdapter
from app.model_scanner import scan_models


def write_hf_folder(root: Path, config: dict) -> None:
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"")


def test_complete_whisper_safetensors_is_runnable(tmp_path: Path):
    write_hf_folder(tmp_path / "whisper", {"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]})

    candidates = HFTransformersASRAdapter().discover(tmp_path)

    assert candidates[0].runnable is True
    assert candidates[0].category == "asr"


def test_complete_unknown_non_text_model_is_probe_required_not_runnable(tmp_path: Path):
    write_hf_folder(tmp_path / "unknown", {"model_type": "custom_encoder", "architectures": ["CustomEncoderModel"]})

    runnable, unsupported = scan_models(tmp_path)

    assert runnable == []
    candidate = next(item for item in unsupported if item.display_name == "unknown")
    assert candidate.category == "asr_probe_required"
    assert candidate.runnable is False


def test_text_llm_safetensors_is_unsupported_llm_format(tmp_path: Path):
    write_hf_folder(tmp_path / "llm", {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "vocab_size": 100, "hidden_size": 8, "num_hidden_layers": 1})

    runnable, unsupported = scan_models(tmp_path)

    assert runnable == []
    candidate = next(item for item in unsupported if item.display_name == "llm")
    assert candidate.category == "unsupported_llm"


def test_structural_unknown_transformer_is_probe_required_not_text_llm(tmp_path: Path):
    write_hf_folder(tmp_path / "unknown-transformer", {"model_type": "unknown_model", "vocab_size": 32000, "hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 16})

    runnable, unsupported = scan_models(tmp_path)

    assert runnable == []
    candidate = next(item for item in unsupported if item.display_name == "unknown-transformer")
    assert candidate.category == "asr_probe_required"
    assert candidate.task == "unknown"


def test_standalone_safetensors_is_incomplete(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"")

    runnable, unsupported = scan_models(tmp_path)

    assert runnable == []
    assert unsupported[0].runnable is False
    assert "config.json" in unsupported[0].missing_files
