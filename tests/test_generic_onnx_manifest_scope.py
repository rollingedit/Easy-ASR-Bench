import json
from pathlib import Path

from app.model_scanner import scan_models


def write_manifest(root: Path, decoding_type: str = "ctc") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.onnx").write_text("", encoding="utf-8")
    (root / "vocab.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    (root / "modelbench.json").write_text(
        json.dumps(
            {
                "schema": "easy_asr_bench.model_manifest.v1",
                "display_name": "CTC model",
                "task": "automatic-speech-recognition",
                "precision": "fp32",
                "files": {"model": "model.onnx"},
                "inputs": {"waveform": {"name": "audio"}},
                "outputs": {"logits": "logits"},
                "decoding": {"type": decoding_type, "blank_token_id": 0, "vocab_file": "vocab.json"},
            }
        ),
        encoding="utf-8",
    )


def test_valid_ctc_manifest_is_runnable(tmp_path: Path):
    write_manifest(tmp_path / "ctc")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in runnable if candidate.adapter_name == "generic_onnx_manifest")
    assert candidate.container_format == "onnx-ctc-manifest"
    assert candidate.runnable is True


def test_seq2seq_manifest_is_recognized_unsupported_with_exact_message(tmp_path: Path):
    write_manifest(tmp_path / "seq2seq", decoding_type="seq2seq")

    runnable, unsupported = scan_models(tmp_path)

    assert not any(candidate.adapter_name == "generic_onnx_manifest" for candidate in runnable)
    candidate = next(candidate for candidate in unsupported if candidate.adapter_name == "generic_onnx_manifest")
    assert any("decoding.type=ctc" in item for item in candidate.missing_files)
    assert "supports only CTC-style ASR ONNX" in candidate.help_text


def test_arbitrary_onnx_without_manifest_requires_modelbench_json(tmp_path: Path):
    (tmp_path / "model.onnx").write_text("", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    candidate = next(candidate for candidate in unsupported if candidate.adapter_name == "generic_onnx_manifest")
    assert "modelbench.json" in candidate.missing_files
    assert candidate.runnable is False


def test_whisper_encoder_decoder_onnx_is_not_generic_runnable(tmp_path: Path):
    model = tmp_path / "whisper-onnx"
    onnx = model / "onnx"
    onnx.mkdir(parents=True)
    (onnx / "encoder_model.onnx").write_text("", encoding="utf-8")
    (onnx / "decoder_model_merged.onnx").write_text("", encoding="utf-8")
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    assert not any(candidate.adapter_name == "generic_onnx_manifest" and candidate.runnable for candidate in runnable)
    assert any(candidate.container_format == "onnx-whisper" for candidate in unsupported)
