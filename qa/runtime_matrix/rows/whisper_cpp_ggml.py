from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.config import load_config
from app.dependency_manager import install_group_for_config, missing_modules_for_config, recovery_command_for_config
from app.model_scanner import scan_models
from qa.runtime_matrix.common import package_versions, sha256, write_row


REPO_ID = "ggerganov/whisper.cpp"
MODEL_FILE = "ggml-tiny.en-q5_1.bin"
MODEL_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/{MODEL_FILE}"
MODEL_SHA256 = "c77c5766f1cef09b6b7d47f21b546cbddd4157886b3b5d6d4f709e91e66c7c2b"


def _download_model(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Easy-ASR-Bench-runtime-matrix"})
    with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def run(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    config = load_config(Path("config.json"))
    config["runtime"]["provider"] = "cpu"
    config["runtime"]["prefer_gpu"] = False
    repair_log = evidence_dir / "whisper_cpp_repair.log"
    repair_command = recovery_command_for_config("whisper_cpp", config)
    missing = missing_modules_for_config("whisper_cpp", config)
    details = {
        "repo_id": REPO_ID,
        "model_file": MODEL_FILE,
        "model_url": MODEL_URL,
        "dependency_versions": package_versions(["pywhispercpp"]),
        "missing_dependency_modules": missing,
        "repair_command": repair_command,
    }
    if missing and install_deps:
        try:
            install_group_for_config("whisper_cpp", Path.cwd(), config, log_path=repair_log)
        except Exception as exc:
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary="whisper.cpp dependency repair failed through the product dependency manager.",
                block_reason=f"{type(exc).__name__}: {exc}",
                external_requirement=repair_command,
                details=details,
                artifacts=[repair_log],
            )
        missing = missing_modules_for_config("whisper_cpp", config)
        details["missing_dependency_modules"] = missing
        details["dependency_versions"] = package_versions(["pywhispercpp"])
    if missing:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="whisper.cpp dependency group is not currently runnable.",
            block_reason=", ".join(missing),
            external_requirement=repair_command,
            details=details,
            artifacts=[repair_log],
        )

    model_path = evidence_dir / "Models" / "whisper_cpp" / MODEL_FILE
    if not model_path.exists():
        if not allow_downloads:
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary="whisper.cpp GGML tiny fixture is not cached locally.",
                block_reason=f"missing local fixture {model_path}",
                external_requirement=f"rerun with --allow-downloads to download {MODEL_URL}",
                details=details,
                artifacts=[repair_log],
            )
        try:
            _download_model(model_path)
        except Exception as exc:
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary="Could not download whisper.cpp GGML tiny fixture.",
                block_reason=f"{type(exc).__name__}: {exc}",
                external_requirement=f"network access to {MODEL_URL}",
                details=details,
                artifacts=[repair_log],
            )
    details["model_path"] = str(model_path)
    details["model_sha256"] = sha256(model_path)
    runnable, unsupported = scan_models(model_path.parent)
    candidates = [candidate for candidate in runnable if candidate.adapter_name == "whisper_cpp"]
    if not candidates:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="whisper.cpp GGML fixture did not scan as a runnable whisper.cpp candidate.",
            details={
                **details,
                "runnable": [candidate.adapter_name for candidate in runnable],
                "unsupported": [{"adapter_name": candidate.adapter_name, "warnings": candidate.warnings, "missing": candidate.missing_files} for candidate in unsupported],
            },
            artifacts=[model_path, repair_log],
        )

    from app.main import adapter_for

    candidate = candidates[0]
    adapter = adapter_for(candidate)
    try:
        adapter.load(candidate, {"provider": "cpu", "prefer_gpu": False, "language": "en", "task": "transcribe"})
        result = adapter.transcribe_chunks(
            [SimpleNamespace(samples=np.zeros(16000, dtype=np.float32))],
            [{"chunk_id": "0001", "start_seconds": 0.0, "end_seconds": 1.0}],
        )
    except Exception as exc:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="whisper.cpp GGML fixture could not load or run through pywhispercpp.",
            block_reason=f"{type(exc).__name__}: {exc}",
            external_requirement="repair whisper_cpp dependency group or provide a compatible pywhispercpp/whisper.cpp runtime",
            details=details,
            artifacts=[model_path, repair_log],
        )
    finally:
        try:
            adapter.unload()
        except Exception:
            pass

    transcript = result.transcript_chunks[0].text if result.transcript_chunks else ""
    return write_row(
        row_id,
        "pass",
        evidence_dir,
        summary="whisper.cpp GGML tiny fixture loaded through pywhispercpp and completed one CPU inference call.",
        details={**details, "transcript": transcript, "metrics": result.metrics, "errors": result.errors},
        artifacts=[model_path, repair_log],
    )
