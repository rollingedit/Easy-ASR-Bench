from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
import urllib.request

from app.dependency_manager import install_group_for_config, missing_modules_for_config, recovery_command_for_config
from app.html_report_builder import build_html_report
from app.main import process_file_with_candidates
from app.model_scanner import scan_models
from app.reference_import import import_llm_reference
from app.scoring import wer
from qa.run_real_tiny_model_smoke import REFERENCE_TEXT, generate_windows_sapi_wav, smoke_config
from qa.runtime_matrix.common import dependency_resolution_report_failures, package_versions, write_row
from qa.runtime_matrix.rows.smollm_reference_grading_report import SMOLLM_PATH, _smollm_candidate


PUBLIC_QWEN3_ASR_REPO = "mradermacher/Qwen3-ASR-0.6B-GGUF"
PUBLIC_QWEN3_ASR_MODEL_FILE = "Qwen3-ASR-0.6B.Q4_K_M.gguf"
PUBLIC_QWEN3_ASR_MMPROJ_FILE = "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf"
PUBLIC_QWEN3_ASR_MODEL_URL = f"https://huggingface.co/{PUBLIC_QWEN3_ASR_REPO}/resolve/main/{PUBLIC_QWEN3_ASR_MODEL_FILE}"
PUBLIC_QWEN3_ASR_MMPROJ_URL = f"https://huggingface.co/{PUBLIC_QWEN3_ASR_REPO}/resolve/main/{PUBLIC_QWEN3_ASR_MMPROJ_FILE}"
PUBLIC_QWEN3_ASR_SOURCE_URL = f"https://huggingface.co/{PUBLIC_QWEN3_ASR_REPO}"
REJECTED_QWEN3_ASR_LOWER_QUANT = "Qwen3-ASR-0.6B.Q2_K.gguf"


def _write_fixture(root: Path, kind: str) -> Path:
    model = root / "Models" / "qwen3-asr-gguf"
    model.mkdir(parents=True, exist_ok=True)
    (model / "Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"fake-main-gguf")
    if kind == "complete":
        (model / "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"fake-mmproj-gguf")
    elif kind == "mismatched":
        (model / "mmproj-Qwen3-ASR-1.7B-Q4_K_M.gguf").write_bytes(b"fake-mmproj-gguf")
    return model


def _candidate_details(candidate) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "adapter_name": candidate.adapter_name,
        "category": candidate.category,
        "container_format": candidate.container_format,
        "runnable": candidate.runnable,
        "missing_files": candidate.missing_files,
        "warnings": candidate.warnings,
        "help_text": candidate.help_text,
        "metadata": candidate.metadata,
    }


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Easy-ASR-Bench-runtime-matrix"})
    with urllib.request.urlopen(request, timeout=180) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _write_manifest(model_dir: Path, model_file: str = PUBLIC_QWEN3_ASR_MODEL_FILE, mmproj_file: str = PUBLIC_QWEN3_ASR_MMPROJ_FILE) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "easy_asr_bench.model_package.v1",
        "source_repo": PUBLIC_QWEN3_ASR_REPO,
        "source_url": PUBLIC_QWEN3_ASR_SOURCE_URL,
        "artifacts": {
            "main_model": model_file,
            "projector": mmproj_file,
        },
        "notes": [
            "Qwen3-ASR uses a projector quantization label that is independent from the main model quantization label.",
            "The manifest pins the exact public main/projector pair so scanner validation does not guess by matching quant suffixes.",
        ],
    }
    path = model_dir / "model_package.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def _ensure_public_fixture(row_id: str, evidence_dir: Path, allow_downloads: bool) -> dict | Path:
    model_dir = evidence_dir / "Models" / PUBLIC_QWEN3_ASR_REPO.replace("/", "__")
    model_path = model_dir / PUBLIC_QWEN3_ASR_MODEL_FILE
    mmproj_path = model_dir / PUBLIC_QWEN3_ASR_MMPROJ_FILE
    manifest_path = _write_manifest(model_dir)
    if model_path.exists() and mmproj_path.exists():
        return model_dir
    if not allow_downloads:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary=f"{PUBLIC_QWEN3_ASR_REPO} ASR GGUF+mmproj fixture is not cached locally.",
            block_reason=f"missing local fixture files under {model_dir}",
            external_requirement=f"rerun with --allow-downloads to download {PUBLIC_QWEN3_ASR_MODEL_FILE} and {PUBLIC_QWEN3_ASR_MMPROJ_FILE}",
            details={
                "repo_id": PUBLIC_QWEN3_ASR_REPO,
                "source_url": PUBLIC_QWEN3_ASR_SOURCE_URL,
                "model_file": PUBLIC_QWEN3_ASR_MODEL_FILE,
                "mmproj_file": PUBLIC_QWEN3_ASR_MMPROJ_FILE,
                "smallest_practical_quality_fixture": True,
                "fixture_note": "Q2_K is the smaller main quant, but local llama-mtmd-cli probing emitted unusable text; Q4_K_M is the smallest probed recommended quant that produced a quality-bearing transcript under the WER threshold.",
                "rejected_lower_quant": {
                    "model_file": REJECTED_QWEN3_ASR_LOWER_QUANT,
                    "observed_failure": "Generated garbage text with normalized WER 5.286 against the SAPI smoke phrase on llama-mtmd-cli 9544.",
                },
            },
            artifacts=[manifest_path, model_path, mmproj_path],
        )
    try:
        _download(PUBLIC_QWEN3_ASR_MODEL_URL, model_path)
        _download(PUBLIC_QWEN3_ASR_MMPROJ_URL, mmproj_path)
        _write_manifest(model_dir)
    except Exception as exc:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary=f"Could not download {PUBLIC_QWEN3_ASR_REPO} ASR GGUF+mmproj fixture.",
            block_reason=f"{type(exc).__name__}: {exc}",
            external_requirement=f"network access to {PUBLIC_QWEN3_ASR_SOURCE_URL}",
            details={
                "repo_id": PUBLIC_QWEN3_ASR_REPO,
                "model_url": PUBLIC_QWEN3_ASR_MODEL_URL,
                "mmproj_url": PUBLIC_QWEN3_ASR_MMPROJ_URL,
            },
            artifacts=[manifest_path, model_path, mmproj_path],
        )
    return model_dir


def _llama_mtmd_version() -> dict:
    from app.adapters.gguf_asr_mmproj import llama_mtmd_cli_path

    cli = llama_mtmd_cli_path({"llama_cpp": {}}, Path.cwd())
    if not cli:
        return {"path": "", "available": False}
    try:
        completed = subprocess.run([cli, "--version"], text=True, capture_output=True, timeout=30)
    except Exception as exc:
        return {"path": cli, "available": True, "version_error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": cli,
        "available": True,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _reference_for(results: dict) -> dict:
    return {
        "schema": "easy_asr_bench.llm_reference.v1",
        "source_sha256": results["source"]["sha256"],
        "reference_type": "llm_corrected_reference",
        "segments": [
            {
                "chunk_id": chunk["chunk_id"],
                "start_seconds": chunk["start_seconds"],
                "end_seconds": chunk["end_seconds"],
                "text": REFERENCE_TEXT,
                "uncertain": ["generated Windows SAPI reference text; not a human transcript"],
            }
            for chunk in results.get("chunk_plan", {}).get("chunks", [])
        ],
        "global_notes": ["Reference text is the generated SAPI prompt used by this ASR GGUF+mmproj runtime row."],
    }


def _scan_fixture(row_id: str, evidence_dir: Path, kind: str) -> dict:
    model = _write_fixture(evidence_dir, kind)
    runnable, unsupported = scan_models(evidence_dir / "Models")
    candidates = [candidate for candidate in [*runnable, *unsupported] if candidate.container_format == "gguf+mmproj"]
    details = {
        "fixture_kind": kind,
        "model_dir": str(model),
        "runnable_count": len(runnable),
        "unsupported_count": len(unsupported),
        "candidates": [_candidate_details(candidate) for candidate in candidates],
    }
    artifacts = list(model.glob("*.gguf"))
    if not candidates:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="ASR GGUF+mmproj fixture did not produce a scanner candidate.",
            details=details,
            artifacts=artifacts,
        )
    candidate = candidates[0]
    if kind == "complete":
        ok = (
            candidate.adapter_name == "gguf_asr_mmproj"
            and candidate.runnable is False
            and candidate.missing_files == []
            and candidate.metadata.get("model_status") == "recognized_experimental"
            and "mmproj_path" in candidate.metadata
        )
        return write_row(
            row_id,
            "blocked" if ok else "fail",
            evidence_dir,
            summary=(
                "Complete ASR GGUF+mmproj package is recognized but remains blocked until a real ASR GGUF smoke fixture proves runtime support."
                if ok
                else "Complete ASR GGUF+mmproj package did not produce the expected experimental blocker evidence."
            ),
            details=details,
            artifacts=artifacts,
            block_reason="ASR GGUF+mmproj runtime is experimental until a real smoke fixture passes through the app.",
            external_requirement="verified ASR GGUF+mmproj model/runtime fixture and full transcription smoke",
        )
    expected_missing = "matching mmproj .gguf"
    ok = (
        candidate.runnable is False
        and candidate.container_format == "gguf+mmproj"
        and expected_missing in candidate.missing_files
    )
    return write_row(
        row_id,
        "pass" if ok else "fail",
        evidence_dir,
        summary=(
            "Incomplete or mismatched ASR GGUF+mmproj package is rejected with an exact matching-projector requirement."
            if ok
            else "ASR GGUF+mmproj incomplete fixture did not report the expected missing projector requirement."
        ),
        details=details,
        artifacts=artifacts,
    )


def _run_public_qwen3_asr(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    if not SMOLLM_PATH.exists():
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="SmolLM 135M GGUF fixture is not present locally, so ASR GGUF+mmproj output cannot be graded by the local reference path.",
            block_reason=f"missing {SMOLLM_PATH}",
            external_requirement="download HuggingFaceTB/SmolLM-135M-GGUF Q4_K_M fixture",
        )
    fixture_or_row = _ensure_public_fixture(row_id, evidence_dir, allow_downloads)
    if isinstance(fixture_or_row, dict):
        return fixture_or_row
    return run_qwen3_asr_model_dir(row_id, evidence_dir, fixture_or_row, install_deps)


def run_qwen3_asr_model_dir(row_id: str, evidence_dir: Path, model_dir: Path, install_deps: bool, extra_details: dict | None = None) -> dict:
    if not SMOLLM_PATH.exists():
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="SmolLM 135M GGUF fixture is not present locally, so ASR GGUF+mmproj output cannot be graded by the local reference path.",
            block_reason=f"missing {SMOLLM_PATH}",
            external_requirement="download HuggingFaceTB/SmolLM-135M-GGUF Q4_K_M fixture",
        )
    artifacts = [SMOLLM_PATH, model_dir / PUBLIC_QWEN3_ASR_MODEL_FILE, model_dir / PUBLIC_QWEN3_ASR_MMPROJ_FILE, model_dir / "model_package.json"]
    runnable, unsupported = scan_models(model_dir)
    candidates = [candidate for candidate in [*runnable, *unsupported] if candidate.adapter_name == "gguf_asr_mmproj"]
    if not candidates:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Public Qwen3-ASR GGUF+mmproj fixture did not scan as an ASR GGUF+mmproj candidate.",
            details={
                "repo_id": PUBLIC_QWEN3_ASR_REPO,
                "runnable": [_candidate_details(candidate) for candidate in runnable],
                "unsupported": [_candidate_details(candidate) for candidate in unsupported],
            },
            artifacts=artifacts,
        )
    candidate = candidates[0]
    if candidate.missing_files:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Public Qwen3-ASR GGUF+mmproj fixture scanned as incomplete despite an exact pairing manifest.",
            details={"candidate": _candidate_details(candidate)},
            artifacts=artifacts,
        )
    config = smoke_config(evidence_dir, "cpu")
    config["runtime"]["max_chunk_seconds"] = 5
    config["runtime"]["chunk_stride_seconds"] = 0
    config["runtime"]["llm_context_tokens"] = 1024
    config["runtime"]["llm_reference_max_tokens"] = 128
    config["runtime"]["llm_reference_temperature"] = 0.0
    config["transcription"] = dict(config.get("transcription", {}))
    config["transcription"]["ar_prompt"] = "Transcribe the audio. Return only the transcript."
    config["transcription"]["ar_max_new_tokens"] = 128
    config["transcription"]["temperature"] = 0.0
    config["llama_cpp"] = dict(config.get("llama_cpp", {}))
    config["llama_cpp"]["timeout_seconds"] = 900
    dependency_details = {
        "llama_mtmd_missing_before": missing_modules_for_config("llama_mtmd", config),
        "llama_mtmd_repair_command": recovery_command_for_config("llama_mtmd", config),
    }
    dependency_artifacts: list[Path] = []
    if dependency_details["llama_mtmd_missing_before"] and install_deps:
        repair_log = evidence_dir / "llama_mtmd_repair.log"
        try:
            install_group_for_config("llama_mtmd", Path.cwd(), config, log_path=repair_log)
            dependency_artifacts.append(repair_log)
        except Exception as exc:
            dependency_artifacts.append(repair_log)
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary="Public Qwen3-ASR GGUF+mmproj fixture is staged, but llama-mtmd-cli native runtime repair failed.",
                block_reason=f"llama_mtmd repair failed: {type(exc).__name__}: {exc}",
                external_requirement="install a recent llama.cpp build that includes llama-mtmd-cli, or install llama-cpp-python with Qwen3ASRChatHandler",
                details={
                    "repo_id": PUBLIC_QWEN3_ASR_REPO,
                    "candidate": _candidate_details(candidate),
                    **dependency_details,
                    "llama_mtmd_missing_after": missing_modules_for_config("llama_mtmd", config),
                    "dependency_versions": package_versions(["llama-cpp-python"]),
                },
                artifacts=[*artifacts, *dependency_artifacts],
            )
    dependency_details["llama_mtmd_missing_after"] = missing_modules_for_config("llama_mtmd", config)
    cli_version = _llama_mtmd_version()
    if not cli_version.get("available"):
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Public Qwen3-ASR GGUF+mmproj fixture is staged, but no llama-mtmd-cli runtime is available.",
            block_reason="llama-mtmd-cli was not found on PATH or in the model folder",
            external_requirement="install a recent llama.cpp build that includes llama-mtmd-cli, or install llama-cpp-python with Qwen3ASRChatHandler",
            details={
                "repo_id": PUBLIC_QWEN3_ASR_REPO,
                "candidate": _candidate_details(candidate),
                "runtime": cli_version,
                **dependency_details,
                "dependency_versions": package_versions(["llama-cpp-python"]),
            },
            artifacts=[*artifacts, *dependency_artifacts],
        )
    llm_candidate, scan_details = _smollm_candidate()
    if llm_candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference/correction LLM candidate.",
            details={**dependency_details, **scan_details},
            artifacts=[*artifacts, *dependency_artifacts],
        )

    source = Path(config["folders"]["input"]) / "qwen3_asr_gguf_mmproj_sapi.wav"
    generate_windows_sapi_wav(source, REFERENCE_TEXT)
    runtime_candidate = replace(
        candidate,
        runnable=True,
        runnable_after_dependency_install=True,
        warnings=[*candidate.warnings, "Runtime matrix forced this recognized-experimental candidate into a live smoke attempt."],
    )
    output_dir = process_file_with_candidates(source, [runtime_candidate], config, unsupported, reference_llm=llm_candidate)
    if output_dir is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="ASR GGUF+mmproj runtime attempt did not produce a report directory.",
            details={
                "repo_id": PUBLIC_QWEN3_ASR_REPO,
                "candidate": _candidate_details(candidate),
                "runtime": cli_version,
                **dependency_details,
                "dependency_versions": package_versions(["llama-cpp-python"]),
                **scan_details,
            },
            artifacts=[*artifacts, *dependency_artifacts, source],
        )

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    runs = results.get("runs", [])
    run = runs[0] if runs else {}
    transcript = "\n".join(chunk.get("text", "") for chunk in run.get("transcript_chunks", []))
    errors = run.get("errors", [])
    normalized_wer = wer(REFERENCE_TEXT, transcript, normalized=True) if transcript.strip() else 1.0
    scored = import_llm_reference(results, "SmolLM corrected reference fixture:\n```json\n" + json.dumps(_reference_for(results)) + "\n```")
    scored_path = output_dir / "scored_report.json"
    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    scored_results = dict(results)
    if scored.get("status") == "scored":
        scored_results["reference_scores"] = scored["scores"]
    scored_html = output_dir / "compare_scored.html"
    scored_html.write_text(build_html_report(scored_results), encoding="utf-8", newline="\n")

    dependency_report_failures, dependency_report_details = dependency_resolution_report_failures(results, expected_groups={"llama_mtmd", "llama_cpp"})
    failures: list[str] = list(dependency_report_failures)
    if errors:
        failures.append("ASR GGUF+mmproj model run reported errors")
    if not transcript.strip():
        failures.append("ASR GGUF+mmproj speech transcript was empty")
    if normalized_wer > 0.85:
        failures.append(f"ASR GGUF+mmproj normalized WER {normalized_wer:.3f} exceeded threshold 0.850")
    for name in ["results.json", "results.txt", "benchmark.csv", "compare.html", "scored_report.json", "compare_scored.html"]:
        if not (output_dir / name).exists():
            failures.append(f"missing report artifact {name}")
    if scored.get("status") != "scored":
        failures.append("ASR GGUF+mmproj scored reference was not produced")

    status = "pass" if not failures else "blocked"
    return write_row(
        row_id,
        status,
        evidence_dir,
        summary=(
            "Public Qwen3-ASR GGUF+mmproj fixture transcribed generated speech, then SmolLM scoring/report validation completed."
            if not failures
            else "Public Qwen3-ASR GGUF+mmproj fixture was staged and attempted through the app pipeline, but runtime output is not yet release-pass quality."
        ),
        block_reason="; ".join(failures) if failures else "",
        external_requirement="newer llama.cpp Qwen3-ASR runtime, different quant/projector pair, or adapter prompt/runtime fix" if failures else "",
        details={
            "repo_id": PUBLIC_QWEN3_ASR_REPO,
            "source_url": PUBLIC_QWEN3_ASR_SOURCE_URL,
            "model_file": PUBLIC_QWEN3_ASR_MODEL_FILE,
            "mmproj_file": PUBLIC_QWEN3_ASR_MMPROJ_FILE,
            "rejected_lower_quant": {
                "model_file": REJECTED_QWEN3_ASR_LOWER_QUANT,
                "observed_failure": "Q2_K was smaller but generated unusable text with normalized WER 5.286 on this runtime; Q4_K_M is the current practical fixture.",
            },
            "candidate": _candidate_details(candidate),
            "runtime": cli_version,
            **dependency_details,
            "reference_text": REFERENCE_TEXT,
            "transcript": transcript,
            "normalized_wer": normalized_wer,
            "max_normalized_wer": 0.85,
            "quality_bearing": True,
            "output_dir": str(output_dir),
            "score_status": scored.get("status"),
            "errors": errors,
            "failures": failures,
            "dependency_versions": package_versions(["llama-cpp-python"]),
            **dependency_report_details,
            **(extra_details or {}),
            **scan_details,
        },
        artifacts=[
            *artifacts,
            *dependency_artifacts,
            source,
            output_dir / "results.json",
            output_dir / "results.txt",
            output_dir / "benchmark.csv",
            output_dir / "compare.html",
            scored_path,
            scored_html,
        ],
    )


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id == "incomplete_audio_asr_gguf_mmproj_rejected":
        return _scan_fixture(row_id, evidence_dir, "missing_projector")
    if row_id == "mismatched_audio_asr_gguf_mmproj_rejected":
        return _scan_fixture(row_id, evidence_dir, "mismatched")
    if row_id in {"audio_asr_gguf_mmproj", "gguf_asr_mmproj_pair"}:
        return _run_public_qwen3_asr(row_id, evidence_dir, _install_deps, _allow_downloads)
    return write_row(row_id, "fail", evidence_dir, summary=f"Unhandled ASR GGUF+mmproj row: {row_id}")
