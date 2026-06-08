from __future__ import annotations

from pathlib import Path

from app.model_scanner import scan_models
from qa.runtime_matrix.common import write_row


def _write(path: Path, data: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8", newline="\n")
    return path


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _create_fixtures(models_root: Path) -> list[Path]:
    artifacts: list[Path] = []
    artifacts.append(_write(models_root / "canary-1b-v2.nemo", "nemo"))

    funasr = models_root / "sensevoice"
    artifacts.extend([_write(funasr / "config.yaml"), _write(funasr / "model.pt")])

    ort = models_root / "moonshine-ort"
    artifacts.extend([_write(ort / "preprocess.ort"), _write(ort / "encode.ort"), _write(ort / "uncached_decode.ort")])

    coreml = models_root / "whisperkit"
    _mkdir(coreml / "AudioEncoder.mlmodelc")
    _mkdir(coreml / "TextDecoder.mlmodelc")

    sherpa = models_root / "sherpa-whisper"
    artifacts.extend([_write(sherpa / "turbo-encoder.onnx"), _write(sherpa / "turbo-decoder.onnx"), _write(sherpa / "turbo-tokens.txt")])

    qwen = models_root / "qwen-onnx"
    for name in ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx", "embed_tokens.bin", "config.json", "preprocessor_config.json", "tokenizer.json"]:
        artifacts.append(_write(qwen / name, "{}" if name.endswith(".json") else ""))

    whisper = models_root / "whisper-onnx" / "onnx"
    artifacts.extend([_write(whisper / "encoder_model_fp16.onnx"), _write(whisper / "decoder_model_merged_fp16.onnx")])
    for name in ["config.json", "preprocessor_config.json", "tokenizer.json"]:
        artifacts.append(_write(models_root / "whisper-onnx" / name, "{}"))

    split = models_root / "split-onnx" / "onnx"
    for name in ["audio_encoder_fp16.onnx", "decoder_model_merged_fp16.onnx", "embed_tokens_fp16.onnx"]:
        artifacts.append(_write(split / name))
    for name in ["config.json", "preprocessor_config.json", "tokenizer.json"]:
        artifacts.append(_write(models_root / "split-onnx" / name, "{}"))
    return artifacts


def _candidate_summary(candidate) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "display_name": candidate.display_name,
        "backend": candidate.backend,
        "container_format": candidate.container_format,
        "category": candidate.category,
        "runnable": candidate.runnable,
        "missing_files": candidate.missing_files,
        "warnings": candidate.warnings,
        "help_text": candidate.help_text,
    }


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    models_root = evidence_dir / "Models"
    artifacts = _create_fixtures(models_root)
    runnable, unsupported = scan_models(models_root)
    summaries = [_candidate_summary(candidate) for candidate in unsupported]
    by_format = {candidate["container_format"]: candidate for candidate in summaries}
    expected = {
        "nemo": {"warning": "NeMo", "missing": []},
        "funasr": {"warning": "FunASR", "missing": ["am.mvn", "tokenizer/BPE file"]},
        "ort": {"warning": "ORT edge", "missing": ["cached_decode.ort"]},
        "mlmodelc": {"warning": "Windows-first", "missing": ["MelSpectrogram.mlmodelc"]},
        "onnx": {"warning": "sherpa-onnx", "backend": "sherpa-onnx"},
        "onnx-qwen-asr": {"warning": "Qwen-style", "missing": ["decoder_weights.data"]},
        "onnx-whisper": {"warning": "Whisper ONNX", "missing": ["onnx/encoder_model_fp16.onnx_data"]},
        "onnx-split-asr": {"warning": "Split audio-encoder", "missing": ["onnx/decoder_model_merged_fp16.onnx_data_1"]},
    }
    failures: list[str] = []
    for container_format, requirement in expected.items():
        candidate = by_format.get(container_format)
        if candidate is None:
            failures.append(f"{container_format}: no unsupported candidate")
            continue
        if candidate["runnable"]:
            failures.append(f"{container_format}: unexpectedly runnable")
        if candidate["category"] != "recognized_unsupported_asr":
            failures.append(f"{container_format}: wrong category {candidate['category']}")
        warning = str(requirement.get("warning", ""))
        text = " ".join(candidate["warnings"] + [candidate["help_text"]])
        if warning and warning not in text:
            failures.append(f"{container_format}: missing warning/help marker {warning}")
        for missing in requirement.get("missing", []):
            if missing not in candidate["missing_files"]:
                failures.append(f"{container_format}: missing expected missing-file entry {missing}")
        backend = requirement.get("backend")
        if backend and candidate["backend"] != backend:
            failures.append(f"{container_format}: expected backend {backend}, got {candidate['backend']}")
    details = {
        "models_root": str(models_root),
        "runnable": [_candidate_summary(candidate) for candidate in runnable],
        "unsupported": summaries,
        "failures": failures,
    }
    return write_row(
        row_id,
        "pass" if not failures and not runnable else "fail",
        evidence_dir,
        summary=(
            "Known unsupported ASR package families are recognized with concrete warnings, missing files, and help text."
            if not failures and not runnable
            else "Known unsupported ASR family recognition did not match expected explanations."
        ),
        details=details,
        artifacts=artifacts,
    )
