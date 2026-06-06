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


def test_complete_unknown_hf_safetensors_is_allowed_to_runtime_probe(tmp_path: Path):
    model = tmp_path / "custom-audio-transformer"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "custom_audio_transformer", "architectures": ["CustomAudioModel"]}),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_text("", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in runnable if candidate.adapter_name == "hf_transformers_asr")
    assert candidate.task == "automatic-speech-recognition"
    assert not any(item.path == model for item in unsupported)


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


def test_faster_whisper_vocabulary_txt_detected(tmp_path: Path):
    model = tmp_path / "faster-whisper"
    model.mkdir()
    (model / "model.bin").write_text("", encoding="utf-8")
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "vocabulary.txt").write_text("", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    assert any(candidate.adapter_name == "faster_whisper" for candidate in runnable)


def test_hf_safetensors_missing_shard_is_reported(tmp_path: Path):
    model = tmp_path / "wav2vec2-sharded"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForCTC"]}), encoding="utf-8")
    (model / "model-00001-of-00002.safetensors").write_text("", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.adapter_name == "hf_transformers_asr")
    assert "model-00002-of-00002.safetensors" in candidate.missing_files


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


def test_asr_gguf_with_mmproj_is_not_text_reference_llm(tmp_path: Path):
    model = tmp_path / "qwen3-asr-gguf"
    model.mkdir()
    (model / "Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"gguf")
    (model / "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"gguf")

    runnable, unsupported = scan_models(tmp_path)

    assert not any(candidate.adapter_name == "gguf_llm_reference" for candidate in unsupported)
    candidate = next(candidate for candidate in unsupported if candidate.container_format == "gguf+mmproj")
    assert candidate.category == "recognized_unsupported_asr"
    assert candidate.missing_files == []


def test_asr_named_gguf_without_mmproj_reports_projector(tmp_path: Path):
    (tmp_path / "Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"gguf")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.container_format == "gguf+mmproj")
    assert "matching mmproj .gguf" in candidate.missing_files
    assert not any(candidate.adapter_name == "gguf_llm_reference" for candidate in unsupported)


def test_nemo_archive_is_recognized_as_unsupported_asr(tmp_path: Path):
    (tmp_path / "canary-1b-v2.nemo").write_bytes(b"nemo")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.container_format == "nemo")
    assert candidate.backend == "nemo"
    assert "NeMo" in candidate.warnings[0]


def test_fun_asr_package_reports_missing_files(tmp_path: Path):
    model = tmp_path / "sensevoice"
    model.mkdir()
    (model / "config.yaml").write_text("", encoding="utf-8")
    (model / "model.pt").write_text("", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.container_format == "funasr")
    assert "am.mvn" in candidate.missing_files
    assert "tokenizer/BPE file" in candidate.missing_files


def test_ort_edge_package_reports_missing_decode_graph(tmp_path: Path):
    model = tmp_path / "moonshine-ort"
    model.mkdir()
    (model / "preprocess.ort").write_text("", encoding="utf-8")
    (model / "encode.ort").write_text("", encoding="utf-8")
    (model / "uncached_decode.ort").write_text("", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.container_format == "ort")
    assert "cached_decode.ort" in candidate.missing_files


def test_coreml_package_is_recognized_as_windows_unsupported(tmp_path: Path):
    model = tmp_path / "whisperkit"
    model.mkdir()
    (model / "AudioEncoder.mlmodelc").mkdir()
    (model / "TextDecoder.mlmodelc").mkdir()

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.container_format == "mlmodelc")
    assert "MelSpectrogram.mlmodelc" in candidate.missing_files
    assert "Windows-first" in candidate.warnings[0]


def test_sherpa_onnx_package_is_recognized(tmp_path: Path):
    model = tmp_path / "sherpa-whisper"
    model.mkdir()
    (model / "turbo-encoder.onnx").write_text("", encoding="utf-8")
    (model / "turbo-decoder.onnx").write_text("", encoding="utf-8")
    (model / "turbo-tokens.txt").write_text("", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.backend == "sherpa-onnx")
    assert candidate.missing_files == []


def test_partial_sherpa_onnx_package_reports_missing_decoder_and_tokens(tmp_path: Path):
    model = tmp_path / "sherpa-whisper"
    model.mkdir()
    (model / "turbo-encoder.onnx").write_text("", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.backend == "sherpa-onnx")
    assert "turbo-decoder.onnx" in candidate.missing_files
    assert "turbo-tokens.txt" in candidate.missing_files


def test_qwen_split_onnx_package_reports_missing_weight_data(tmp_path: Path):
    model = tmp_path / "qwen-onnx"
    model.mkdir()
    for name in ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx", "embed_tokens.bin", "config.json", "preprocessor_config.json", "tokenizer.json"]:
        (model / name).write_text("", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.container_format == "onnx-qwen-asr")
    assert "decoder_weights.data" in candidate.missing_files


def test_whisper_onnx_package_is_recognized_without_generic_noise(tmp_path: Path):
    model = tmp_path / "whisper-onnx"
    onnx = model / "onnx"
    onnx.mkdir(parents=True)
    (onnx / "encoder_model_fp16.onnx").write_text("", encoding="utf-8")
    (onnx / "decoder_model_merged_fp16.onnx").write_text("", encoding="utf-8")
    for name in ["config.json", "preprocessor_config.json", "tokenizer.json"]:
        (model / name).write_text("{}", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.container_format == "onnx-whisper")
    assert "onnx/encoder_model_fp16.onnx_data" in candidate.missing_files
    assert not any(candidate.container_format == "onnx" and candidate.adapter_name == "generic_onnx_manifest" for candidate in unsupported)


def test_partial_whisper_onnx_package_reports_missing_decoder(tmp_path: Path):
    model = tmp_path / "whisper-onnx"
    onnx = model / "onnx"
    onnx.mkdir(parents=True)
    (onnx / "encoder_model_fp16.onnx").write_text("", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.container_format == "onnx-whisper")
    assert "decoder_model_merged*.onnx or decoder_with_past_model*.onnx" in candidate.missing_files
    assert not any(candidate.container_format == "onnx" and candidate.adapter_name == "generic_onnx_manifest" for candidate in unsupported)


def test_split_audio_encoder_onnx_package_reports_all_sidecars(tmp_path: Path):
    model = tmp_path / "split-onnx"
    onnx = model / "onnx"
    onnx.mkdir(parents=True)
    for name in ["audio_encoder_fp16.onnx", "decoder_model_merged_fp16.onnx", "embed_tokens_fp16.onnx"]:
        (onnx / name).write_text("", encoding="utf-8")
    for name in ["config.json", "preprocessor_config.json", "tokenizer.json"]:
        (model / name).write_text("{}", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.container_format == "onnx-split-asr")
    assert "onnx/decoder_model_merged_fp16.onnx_data_1" in candidate.missing_files


def test_partial_split_audio_encoder_onnx_package_reports_missing_graphs(tmp_path: Path):
    model = tmp_path / "split-onnx"
    onnx = model / "onnx"
    onnx.mkdir(parents=True)
    (onnx / "audio_encoder_fp16.onnx").write_text("", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.container_format == "onnx-split-asr")
    assert "decoder_model_merged*.onnx" in candidate.missing_files
    assert "embed_tokens*.onnx" in candidate.missing_files
