import json
from pathlib import Path

from app.model_scanner import scan_models
from app.results_writer import candidate_to_dict


def test_fp32_safetensors_detected_and_runtime_supported_when_complete(tmp_path: Path):
    model = tmp_path / "fp32-asr"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForCTC"], "torch_dtype": "float32"}), encoding="utf-8")
    (model / "model.safetensors").write_text("", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")

    runnable, _ = scan_models(tmp_path)
    data = candidate_to_dict(next(candidate for candidate in runnable if candidate.adapter_name == "hf_transformers_asr"))

    assert data["detected_precision"] == "fp32"
    assert data["runtime_precision_supported"] is True


def test_bf16_safetensors_detected_and_runtime_supported_when_complete(tmp_path: Path):
    model = tmp_path / "bf16-asr"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForCTC"], "torch_dtype": "bfloat16"}), encoding="utf-8")
    (model / "model.safetensors").write_text("", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")

    runnable, _ = scan_models(tmp_path)
    data = candidate_to_dict(next(candidate for candidate in runnable if candidate.adapter_name == "hf_transformers_asr"))

    assert data["detected_precision"] == "bfloat16"
    assert data["precision_bucket"] == "16-bit / BF16"
    assert data["runtime_precision_supported"] is True


def test_unsupported_onnx_quant_folder_reports_detected_precision_without_runtime_claim(tmp_path: Path):
    model = tmp_path / "onnx-q4"
    q4 = model / "q4"
    q4.mkdir(parents=True)
    (q4 / "encoder.onnx").write_text("", encoding="utf-8")

    _, unsupported = scan_models(tmp_path)
    data = candidate_to_dict(next(candidate for candidate in unsupported if candidate.container_format == "onnx-qwen-asr"))

    assert data["runtime_precision_supported"] is False
    assert data["detected_precision"].lower() == "q4"
    assert data["precision_bucket"] == "4-bit / Q4"
    assert "runtime support depends" in data["runtime_precision_reason"]


def test_gguf_q4_is_reference_llm_not_asr_runtime(tmp_path: Path):
    (tmp_path / "reference.Q4_K_M.gguf").write_bytes(b"gguf")

    _, unsupported = scan_models(tmp_path)
    candidate = next(candidate for candidate in unsupported if candidate.adapter_name == "gguf_llm_reference")
    data = candidate_to_dict(candidate)

    assert data["detected_precision"].lower() == "q4_k_m"
    assert data["runtime_precision_supported"] is False
    assert candidate.category == "reference_llm"
    assert candidate.task == "llm-corrected-reference"
