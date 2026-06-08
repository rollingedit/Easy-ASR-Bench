from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

from app.config import load_config
from app.adapters.openai_whisper_pt import KNOWN_OFFICIAL_SHA256, OpenAIWhisperPTAdapter, is_verified_official_checkpoint
from app.dependency_manager import install_group_for_config, missing_modules_for_config, recovery_command_for_config
from app.html_report_builder import build_html_report
from app.main import process_file_with_candidates
from app.model_scanner import scan_models
from app.model_status import model_status_label
from app.reference_import import import_llm_reference
from app.scoring import wer
from qa.runtime_matrix.common import dependency_resolution_report_failures, sha256, write_row
from qa.runtime_matrix.rows.smollm_reference_grading_report import SMOLLM_PATH, _smollm_candidate
from qa.run_real_tiny_model_smoke import REFERENCE_TEXT, generate_windows_sapi_wav, smoke_config


TINY_PT = "tiny.pt"
TINY_PT_SHA256 = KNOWN_OFFICIAL_SHA256[TINY_PT]
TINY_PT_URL = f"https://openaipublic.azureedge.net/main/whisper/models/{TINY_PT_SHA256}/{TINY_PT}"
MAX_NORMALIZED_WER = 0.85


def _write_fake_checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"easy-asr-bench fake unsafe pt checkpoint")
    return path


def _find_candidate(candidates: list, path: Path):
    for candidate in candidates:
        if candidate.path.resolve() == path.resolve():
            return candidate
    return None


def _download_official_checkpoint(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(TINY_PT_URL, headers={"User-Agent": "Easy-ASR-Bench-runtime-matrix"})
    with urllib.request.urlopen(request, timeout=240) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(destination)


def _reference_for(results: dict, text: str) -> dict:
    return {
        "schema": "easy_asr_bench.llm_reference.v1",
        "source_sha256": results["source"]["sha256"],
        "reference_type": "llm_corrected_reference",
        "segments": [
            {
                "chunk_id": chunk["chunk_id"],
                "start_seconds": chunk["start_seconds"],
                "end_seconds": chunk["end_seconds"],
                "text": text,
                "uncertain": ["generated Windows SAPI reference text; not a human transcript"],
            }
            for chunk in results.get("chunk_plan", {}).get("chunks", [])
        ],
        "global_notes": ["Reference text is the generated SAPI prompt used by this runtime row."],
    }


def _write_scored_artifacts(results: dict, output_dir: Path, reference_text: str) -> tuple[dict, Path]:
    reference = _reference_for(results, reference_text)
    scored = import_llm_reference(results, "SmolLM corrected reference fixture:\n```json\n" + json.dumps(reference) + "\n```")
    scored_path = output_dir / "scored_report.json"
    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    scored_results = dict(results)
    if scored.get("status") == "scored":
        scored_results["reference_scores"] = scored["scores"]
    scored_html = output_dir / "compare_scored.html"
    scored_html.write_text(build_html_report(scored_results), encoding="utf-8", newline="\n")
    return scored, scored_html


def _blocked_pt_row(row_id: str, evidence_dir: Path, filename: str) -> dict:
    models_root = evidence_dir / "Models"
    checkpoint = _write_fake_checkpoint(models_root / filename)
    runnable, unsupported = scan_models(models_root)
    candidate = _find_candidate([*runnable, *unsupported], checkpoint)
    details = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "runnable_count": len(runnable),
        "unsupported_count": len(unsupported),
    }
    if candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary=f"{filename} was not discovered by the OpenAI Whisper .pt scanner.",
            details=details,
            artifacts=[checkpoint],
        )
    details.update(
        {
            "candidate_id": candidate.candidate_id,
            "adapter_name": candidate.adapter_name,
            "runnable": candidate.runnable,
            "model_status": model_status_label(candidate),
            "warnings": candidate.warnings,
            "help_text": candidate.help_text,
            "verified_official_checkpoint": is_verified_official_checkpoint(checkpoint),
        }
    )
    try:
        OpenAIWhisperPTAdapter().load(candidate, {"security": {"allow_pickle_or_pt_files": False}})
    except RuntimeError as exc:
        details["load_guard_error"] = str(exc)
    else:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary=f"{filename} unexpectedly loaded with unsafe .pt loading disabled.",
            details=details,
            artifacts=[checkpoint],
        )
    required_markers = [
        candidate.adapter_name == "openai_whisper_pt",
        candidate.runnable is False,
        model_status_label(candidate) == "Unsafe blocked",
        any("pickle" in warning.lower() for warning in candidate.warnings),
        "Blocked .pt checkpoint" in details.get("load_guard_error", ""),
    ]
    if filename.lower() in KNOWN_OFFICIAL_SHA256:
        required_markers.append(any("filenames are not trusted" in warning for warning in candidate.warnings))
    return write_row(
        row_id,
        "pass" if all(required_markers) else "fail",
        evidence_dir,
        summary=(
            f"{filename} is blocked by scanner classification and runtime load guard with unsafe .pt loading disabled."
            if all(required_markers)
            else f"{filename} did not produce the expected blocked .pt safety evidence."
        ),
        details=details,
        artifacts=[checkpoint],
    )


def _checksum_verified_row(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    config = smoke_config(evidence_dir, "cpu")
    details = {
        "official_allowlist_count": len(KNOWN_OFFICIAL_SHA256),
        "tiny_pt_sha256": TINY_PT_SHA256,
        "tiny_pt_url": TINY_PT_URL,
        "download_supported_by_row": True,
        "dependency_repair_command": recovery_command_for_config("openai_whisper", config),
    }
    checkpoint = evidence_dir / "Models" / "openai_whisper" / TINY_PT
    details["checkpoint"] = str(checkpoint)
    if not checkpoint.exists() and not allow_downloads:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="OpenAI Whisper .pt checksum-verified runtime proof requires an actual official checkpoint file.",
            details=details,
            block_reason="official OpenAI Whisper .pt checkpoint is not present and downloads are disabled",
            external_requirement="download official tiny.pt or provide a cached official checkpoint, then rerun the row",
        )
    if not checkpoint.exists():
        try:
            _download_official_checkpoint(checkpoint)
        except Exception as exc:
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary="Could not download official OpenAI Whisper tiny.pt checkpoint.",
                block_reason=f"{type(exc).__name__}: {exc}",
                external_requirement=f"network access to {TINY_PT_URL}",
                details=details,
            )
    details["checkpoint_sha256"] = sha256(checkpoint)
    if not is_verified_official_checkpoint(checkpoint):
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Downloaded OpenAI Whisper tiny.pt did not match the official SHA256 allowlist.",
            details=details,
            artifacts=[checkpoint],
        )

    missing = missing_modules_for_config("openai_whisper", config)
    details["missing_dependency_modules"] = missing
    if missing and install_deps:
        repair_log = evidence_dir / "openai_whisper_repair.log"
        try:
            install_group_for_config("openai_whisper", Path.cwd(), config, log_path=repair_log)
        except Exception as exc:
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary="OpenAI Whisper dependency repair failed through the product dependency manager.",
                block_reason=f"{type(exc).__name__}: {exc}",
                external_requirement=details["dependency_repair_command"],
                details=details,
                artifacts=[checkpoint, repair_log],
            )
        missing = missing_modules_for_config("openai_whisper", config)
        details["missing_dependency_modules"] = missing
        details["repair_log"] = str(repair_log)
    if missing:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="OpenAI Whisper dependency group is not currently runnable.",
            block_reason=", ".join(missing),
            external_requirement=details["dependency_repair_command"],
            details=details,
            artifacts=[checkpoint],
        )
    if not SMOLLM_PATH.exists():
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="SmolLM 135M GGUF fixture is not present locally, so verified OpenAI Whisper .pt output cannot be graded by the local reference path.",
            block_reason=f"missing {SMOLLM_PATH}",
            external_requirement="download HuggingFaceTB/SmolLM-135M-GGUF Q4_K_M fixture",
            details=details,
            artifacts=[checkpoint],
        )
    try:
        import llama_cpp  # noqa: F401
    except ModuleNotFoundError:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="llama-cpp-python is not installed, so SmolLM GGUF cannot run after verified OpenAI Whisper .pt ASR.",
            block_reason="missing llama_cpp import",
            external_requirement="install llama_cpp dependency group",
            details=details,
            artifacts=[checkpoint, SMOLLM_PATH],
        )

    runnable, unsupported = scan_models(checkpoint.parent)
    candidate = _find_candidate(runnable, checkpoint)
    if candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Verified OpenAI Whisper tiny.pt was not discovered as a runnable .pt candidate.",
            details={
                **details,
                "runnable": [item.adapter_name for item in runnable],
                "unsupported": [{"adapter_name": item.adapter_name, "warnings": item.warnings, "missing": item.missing_files} for item in unsupported],
            },
            artifacts=[checkpoint, SMOLLM_PATH],
        )

    llm_candidate, scan_details = _smollm_candidate()
    if llm_candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference/correction LLM candidate.",
            details={**details, **scan_details},
            artifacts=[checkpoint, SMOLLM_PATH],
        )

    source = evidence_dir / "Input" / "openai_whisper_tiny_pt_sapi.wav"
    generate_windows_sapi_wav(source, REFERENCE_TEXT)
    config["runtime"]["provider"] = "cpu"
    config["runtime"]["prefer_gpu"] = False
    config["security"] = dict(config.get("security", {}))
    config["security"]["allow_pickle_or_pt_files"] = False
    output_dir = process_file_with_candidates(source, [candidate], config, unsupported, reference_llm=llm_candidate)
    if output_dir is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Verified OpenAI Whisper tiny.pt run did not produce a report directory.",
            details={**details, **scan_details},
            artifacts=[checkpoint, SMOLLM_PATH, source],
        )
    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    transcript = "\n".join(chunk.get("text", "") for chunk in results["runs"][0].get("transcript_chunks", []))
    normalized_wer = wer(REFERENCE_TEXT, transcript, normalized=True) if transcript.strip() else 1.0
    scored, scored_html = _write_scored_artifacts(results, output_dir, REFERENCE_TEXT)
    run_id = results["runs"][0]["model"]["candidate_id"]
    score = scored.get("scores", {}).get(run_id, {})
    dependency_report_failures, dependency_report_details = dependency_resolution_report_failures(results, expected_groups={"openai_whisper", "llama_cpp"})
    failures: list[str] = list(dependency_report_failures)
    if not transcript.strip():
        failures.append("verified OpenAI Whisper .pt transcript was empty")
    if normalized_wer > MAX_NORMALIZED_WER:
        failures.append(f"normalized WER {normalized_wer:.3f} exceeded threshold {MAX_NORMALIZED_WER:.3f}")
    for name in ["results.json", "results.txt", "benchmark.csv", "compare.html", "scored_report.json", "compare_scored.html"]:
        if not (output_dir / name).exists():
            failures.append(f"missing report artifact {name}")
    if scored.get("status") != "scored" or score.get("normalized_wer") is None:
        failures.append("verified OpenAI Whisper .pt scored reference was not produced")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "Official OpenAI Whisper tiny.pt was SHA256-verified, ran through the app pipeline, then SmolLM grading/report validation completed."
            if not failures
            else "Official OpenAI Whisper tiny.pt validation failed."
        ),
        details={
            **details,
            **scan_details,
            "transcript": transcript,
            "reference_text": REFERENCE_TEXT,
            "normalized_wer": normalized_wer,
            "max_normalized_wer": MAX_NORMALIZED_WER,
            "output_dir": str(output_dir),
            "score_status": scored.get("status"),
            "openai_whisper_pt_score": {
                "candidate_id": run_id,
                "normalized_wer": score.get("normalized_wer"),
                "balanced_score": score.get("balanced_score"),
                "balanced_rank": score.get("balanced_rank"),
                "alignment_mode": score.get("alignment_mode"),
            },
            **dependency_report_details,
            "failures": failures,
        },
        artifacts=[
            checkpoint,
            SMOLLM_PATH,
            source,
            output_dir / "results.json",
            output_dir / "results.txt",
            output_dir / "benchmark.csv",
            output_dir / "compare.html",
            output_dir / "scored_report.json",
            scored_html,
        ],
    )


def run(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    if row_id == "openai_whisper_pt_unknown_blocked":
        return _blocked_pt_row(row_id, evidence_dir, "custom.pt")
    if row_id == "openai_pt_unverified_blocked":
        return _blocked_pt_row(row_id, evidence_dir, "tiny.pt")
    if row_id == "openai_whisper_pt_checksum_verified":
        return _checksum_verified_row(row_id, evidence_dir, install_deps, allow_downloads)
    return write_row(row_id, "fail", evidence_dir, summary=f"Unhandled OpenAI Whisper .pt safety row: {row_id}")
