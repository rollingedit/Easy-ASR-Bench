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


def test_hf_safetensors_float32_detected(tmp_path: Path):
    model = tmp_path / "wav2vec2-float32"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForCTC"], "torch_dtype": "float32"}),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_text("", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in runnable if candidate.adapter_name == "hf_transformers_asr")
    assert candidate.precision == "fp32"
    assert candidate.quantization_label == "32-bit / FP32"


def test_hf_safetensors_bfloat16_detected(tmp_path: Path):
    model = tmp_path / "wav2vec2-bf16"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForCTC"], "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_text("", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in runnable if candidate.adapter_name == "hf_transformers_asr")
    assert candidate.precision == "bfloat16"
    assert candidate.quantization_label == "16-bit / BF16"


def test_whisper_cpp_detected(tmp_path: Path):
    (tmp_path / "ggml-base.bin").write_text("", encoding="utf-8")
    runnable, unsupported = scan_models(tmp_path)
    assert any(candidate.adapter_name == "whisper_cpp" for candidate in runnable)


def test_whisper_cpp_f32_precision_detected(tmp_path: Path):
    (tmp_path / "ggml-large-f32.bin").write_text("", encoding="utf-8")
    runnable, unsupported = scan_models(tmp_path)
    candidate = next(candidate for candidate in runnable if candidate.adapter_name == "whisper_cpp")
    assert candidate.quantization_label == "32-bit / FP32"


def test_multifile_onnx_float32_folder_detected(tmp_path: Path):
    model = tmp_path / "known-onnx"
    precision = model / "float32"
    precision.mkdir(parents=True)
    for name in [
        "encoder.onnx",
        "encoder.onnx_data",
        "prompt_encode.onnx",
        "prompt_encode.onnx_data",
        "decode_step.onnx",
        "decode_step.onnx_data",
        "embed_tokens.onnx",
        "embed_tokens.onnx_data",
    ]:
        (precision / name).write_text("", encoding="utf-8")
    for name in ["tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"]:
        (model / name).write_text("{}", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in runnable if candidate.adapter_name == "granite_onnx_ar")
    assert candidate.precision == "float32"
    assert candidate.quantization_label == "32-bit / FP32"


def test_faster_whisper_fp32_alias_detected(tmp_path: Path):
    model = tmp_path / "faster-whisper-fp32"
    model.mkdir()
    (model / "model.bin").write_text("", encoding="utf-8")
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in runnable if candidate.adapter_name == "faster_whisper")
    assert candidate.precision == "fp32"
    assert candidate.quantization_label == "32-bit / FP32"


def test_standalone_safetensors_explained(tmp_path: Path):
    (tmp_path / "random.safetensors").write_text("", encoding="utf-8")
    runnable, unsupported = scan_models(tmp_path)
    assert not runnable
    assert any("complete Hugging Face" in candidate.help_text or candidate.missing_files for candidate in unsupported)


def test_gguf_reference_llm_is_complete_single_file(tmp_path: Path):
    (tmp_path / "reference.Q4_K_M.gguf").write_bytes(b"gguf")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.adapter_name == "gguf_llm_reference")
    assert candidate.category == "reference_llm"
    assert candidate.missing_files == []
    assert candidate.runnable_after_dependency_install is True


def test_text_llm_safetensors_explains_gguf_requirement(tmp_path: Path):
    model = tmp_path / "mistral-safetensors"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "mistral", "architectures": ["MistralForCausalLM"], "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_text("", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.category == "unsupported_llm")
    assert candidate.task == "text-generation"
    assert candidate.quantization_label == "16-bit / BF16"
    assert "GGUF export (.gguf) for local reference LLM loading" in candidate.missing_files
    assert "GGUF-only" in candidate.warnings[0]


def test_custom_causallm_safetensors_explains_gguf_requirement(tmp_path: Path):
    model = tmp_path / "custom-llm"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "new_family", "architectures": ["NewFamilyForCausalLM"], "torch_dtype": "float16"}),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_text("", encoding="utf-8")
    (model / "tokenizer.model").write_text("", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.category == "unsupported_llm")
    assert candidate.display_name == "custom-llm"
    assert "GGUF export (.gguf) for local reference LLM loading" in candidate.missing_files


def test_structural_non_asr_transformer_safetensors_explains_gguf_requirement(tmp_path: Path):
    model = tmp_path / "unknown-transformer"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "unknown_model", "vocab_size": 32000, "hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 16}),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_text("", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.category == "unsupported_llm")
    assert candidate.display_name == "unknown-transformer"
    assert candidate.task == "text-generation"
