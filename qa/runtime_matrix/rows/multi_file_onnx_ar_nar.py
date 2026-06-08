from __future__ import annotations

import json
from pathlib import Path

from app.adapters.granite_onnx_ar import REQUIRED_AR_FILES
from app.adapters.granite_onnx_nar import REQUIRED_NAR_FILES
from app.model_scanner import scan_models
from qa.runtime_matrix.common import write_row


def _touch(path: Path, payload: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_shared_metadata(root: Path) -> list[Path]:
    artifacts = [
        _touch(root / "tokenizer.json", json.dumps({"model": {"type": "BPE"}, "version": "1.0"}).encode("utf-8")),
        _touch(root / "tokenizer_config.json", b'{"model_max_length": 128}\n'),
        _touch(root / "preprocessor_config.json", b'{"sampling_rate": 16000}\n'),
    ]
    return artifacts


def _write_complete_package(root: Path, precision: str, required_files: list[str]) -> list[Path]:
    artifacts = _write_shared_metadata(root)
    artifacts.extend(_touch(root / precision / name) for name in required_files)
    return artifacts


def _write_incomplete_package(root: Path, precision: str, required_files: list[str], omitted: set[str]) -> list[Path]:
    artifacts = _write_shared_metadata(root)
    artifacts.extend(_touch(root / precision / name) for name in required_files if name not in omitted)
    return artifacts


def _candidate_summary(candidate) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "adapter_name": candidate.adapter_name,
        "display_name": candidate.display_name,
        "container_format": candidate.container_format,
        "precision": candidate.precision,
        "quantization_label": candidate.quantization_label,
        "runnable": candidate.runnable,
        "missing_files": list(candidate.missing_files),
        "warnings": list(candidate.warnings),
    }


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    models_root = evidence_dir / "Models"
    artifacts: list[Path] = []
    artifacts.extend(_write_complete_package(models_root / "granite-ar-complete", "fp32", REQUIRED_AR_FILES))
    artifacts.extend(_write_complete_package(models_root / "granite-nar-complete", "fp16w", REQUIRED_NAR_FILES))
    artifacts.extend(
        _write_incomplete_package(
            models_root / "granite-ar-incomplete",
            "int8",
            REQUIRED_AR_FILES,
            {"decode_step.onnx_data", "embed_tokens.onnx_data"},
        )
    )
    artifacts.extend(
        _write_incomplete_package(
            models_root / "granite-nar-incomplete",
            "f32",
            REQUIRED_NAR_FILES,
            {"editor.onnx_data", "embed_tokens.onnx_data"},
        )
    )

    runnable, unsupported = scan_models(models_root)
    runnable_summary = [_candidate_summary(candidate) for candidate in runnable]
    unsupported_summary = [_candidate_summary(candidate) for candidate in unsupported]
    def normalized_id(item: dict) -> str:
        return str(item["candidate_id"]).replace("-", "_")

    ar_complete = [item for item in runnable_summary if item["adapter_name"] == "granite_onnx_ar" and "granite_ar_complete" in normalized_id(item)]
    nar_complete = [item for item in runnable_summary if item["adapter_name"] == "granite_onnx_nar" and "granite_nar_complete" in normalized_id(item)]
    ar_incomplete = [item for item in unsupported_summary if item["adapter_name"] == "granite_onnx_ar" and "granite_ar_incomplete" in normalized_id(item)]
    nar_incomplete = [item for item in unsupported_summary if item["adapter_name"] == "granite_onnx_nar" and "granite_nar_incomplete" in normalized_id(item)]

    failures = []
    if not ar_complete:
        failures.append("complete AR package did not scan as runnable granite_onnx_ar")
    if not nar_complete:
        failures.append("complete NAR package did not scan as runnable granite_onnx_nar")
    if not ar_incomplete:
        failures.append("incomplete AR package did not scan as unsupported granite_onnx_ar")
    elif not {"int8/decode_step.onnx_data", "int8/embed_tokens.onnx_data"} <= set(ar_incomplete[0]["missing_files"]):
        failures.append("incomplete AR package did not report exact missing AR sidecars")
    if not nar_incomplete:
        failures.append("incomplete NAR package did not scan as unsupported granite_onnx_nar")
    elif not {"f32/editor.onnx_data", "f32/embed_tokens.onnx_data"} <= set(nar_incomplete[0]["missing_files"]):
        failures.append("incomplete NAR package did not report exact missing NAR sidecars")
    false_positive_split_onnx = [
        item
        for item in unsupported_summary
        if item["container_format"] in {"onnx-qwen-asr", "onnx-split-asr", "onnx-whisper"}
        and item["candidate_id"].split("__")[-1] in {"fp32", "fp16w", "int8", "f32"}
    ]
    if false_positive_split_onnx:
        failures.append("runnable AR/NAR precision subfolders were also reported as unsupported split ONNX packages")

    details = {
        "runnable": runnable_summary,
        "unsupported": unsupported_summary,
        "runtime_limit": "Fixture files are structural placeholders. Full AR/NAR inference still requires a tiny real compatible multi-file ONNX model.",
        "false_positive_split_onnx": false_positive_split_onnx,
        "failures": failures,
    }
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "Multi-file ONNX AR/NAR packages are scanned with exact runnable and incomplete sidecar behavior."
            if not failures
            else "Multi-file ONNX AR/NAR scanner evidence failed."
        ),
        details=details,
        artifacts=artifacts,
    )
