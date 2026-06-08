from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.html_report_builder import build_html_report
from app.main import process_file_with_candidates
from app.model_scanner import scan_models
from app.reference_import import import_llm_reference
from app.results_writer import build_results, write_all_reports
from app.scoring import wer
from qa.run_real_tiny_model_smoke import REFERENCE_TEXT, generate_windows_sapi_wav, smoke_config
from qa.runtime_matrix.common import dependency_resolution_report_failures, package_versions, sha256, write_row
from qa.runtime_matrix.rows.real_public_media_faster_whisper_smollm import _download_fixture
from qa.runtime_matrix.rows.smollm_reference_grading_report import SMOLLM_PATH, _smollm_candidate
from qa.runtime_matrix.rows.whisper_cpp_ggml import MODEL_FILE, MODEL_SHA256, MODEL_URL, REPO_ID, _download_model


def _reference_for(
    results: dict,
    text: str,
    uncertain_note: str = "whisper.cpp blank-audio tiny fixture; not a quality-bearing human reference",
    global_note: str = "Reference text is derived from the tiny whisper.cpp fixture output for stable structural scoring only.",
) -> dict:
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
                "uncertain": [uncertain_note],
            }
            for chunk in results.get("chunk_plan", {}).get("chunks", [])
        ],
        "global_notes": [global_note],
    }


def _find_cached_model(filename: str, expected_sha256: str) -> Path | None:
    for root_name in ("Temp", "Models", "Cache"):
        root = Path.cwd() / root_name
        if not root.exists():
            continue
        for candidate in root.rglob(filename):
            if not candidate.is_file():
                continue
            digest = sha256(candidate).removeprefix("sha256:").lower()
            if digest == expected_sha256.lower():
                return candidate
    return None


def _ensure_whisper_cpp_fixture(evidence_dir: Path, row_id: str, allow_downloads: bool, artifacts: list[Path] | None = None) -> dict | Path:
    model_path = evidence_dir / "Models" / "whisper_cpp" / MODEL_FILE
    if model_path.exists() and sha256(model_path).removeprefix("sha256:").lower() == MODEL_SHA256:
        return model_path
    cached = _find_cached_model(MODEL_FILE, MODEL_SHA256)
    if cached is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, model_path)
        return model_path
    if not allow_downloads:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="whisper.cpp GGML tiny fixture is not cached locally.",
            block_reason=f"missing local fixture {model_path}",
            external_requirement=f"rerun with --allow-downloads to download {MODEL_URL}",
            details={"repo_id": REPO_ID, "model_file": MODEL_FILE},
            artifacts=artifacts or [SMOLLM_PATH],
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
            details={"repo_id": REPO_ID, "model_file": MODEL_FILE},
            artifacts=artifacts or [SMOLLM_PATH],
        )
    if sha256(model_path).removeprefix("sha256:").lower() != MODEL_SHA256:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Downloaded whisper.cpp GGML tiny fixture did not match the expected SHA256.",
            details={"repo_id": REPO_ID, "model_file": MODEL_FILE, "expected_sha256": MODEL_SHA256, "actual_sha256": sha256(model_path)},
            artifacts=artifacts or [SMOLLM_PATH, model_path],
        )
    return model_path


def _run_public_media_quality(row_id: str, evidence_dir: Path, allow_downloads: bool) -> dict:
    model_path_or_row = _ensure_whisper_cpp_fixture(evidence_dir, row_id, allow_downloads, [SMOLLM_PATH])
    if isinstance(model_path_or_row, dict):
        return model_path_or_row
    model_path = model_path_or_row
    fixture_id = "wikimedia_public_domain_spoken_words_webm" if row_id == "real_public_video_whisper_cpp_ggml_smollm_grading" else "wikimedia_cc0_word_wav"
    source, fixture_details, fixture_error = _download_fixture(fixture_id, evidence_dir, allow_downloads)
    if fixture_error or source is None:
        return write_row(
            row_id,
            "blocked" if not allow_downloads else "fail",
            evidence_dir,
            summary="Real public media fixture is not available for whisper.cpp ASR+SmolLM validation.",
            block_reason=fixture_error if not allow_downloads else None,
            external_requirement="rerun with --allow-downloads after source/license review" if not allow_downloads else None,
            details=fixture_details,
            artifacts=[SMOLLM_PATH, model_path],
        )
    config = smoke_config(evidence_dir, "cpu")
    config["runtime"]["llm_context_tokens"] = 1024
    config["runtime"]["llm_reference_max_tokens"] = 128
    config["runtime"]["llm_reference_temperature"] = 0.0
    runnable, unsupported = scan_models(model_path.parent)
    candidates = [candidate for candidate in runnable if candidate.adapter_name == "whisper_cpp"]
    if not candidates:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="whisper.cpp GGML fixture did not scan as a runnable whisper.cpp candidate.",
            details={
                **fixture_details,
                "repo_id": REPO_ID,
                "model_file": MODEL_FILE,
                "runnable": [candidate.adapter_name for candidate in runnable],
                "unsupported": [{"adapter_name": candidate.adapter_name, "warnings": candidate.warnings, "missing": candidate.missing_files} for candidate in unsupported],
            },
            artifacts=[SMOLLM_PATH, model_path, source],
        )
    llm_candidate, scan_details = _smollm_candidate()
    if llm_candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference/correction LLM candidate.",
            details={**fixture_details, **scan_details},
            artifacts=[SMOLLM_PATH, model_path, source],
        )

    output_dir = process_file_with_candidates(source, [candidates[0]], config, unsupported, reference_llm=llm_candidate)
    if output_dir is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="whisper.cpp real public media row did not produce a report directory.",
            details={**fixture_details, **scan_details},
            artifacts=[SMOLLM_PATH, model_path, source],
        )
    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    runs = results.get("runs", [])
    transcript = "\n".join(chunk.get("text", "") for run in runs for chunk in run.get("transcript_chunks", []))
    expected_text = fixture_details["fixture"].get("expected_text") or ""
    normalized_wer = wer(expected_text, transcript, normalized=True) if expected_text and transcript.strip() else 1.0
    reference = _reference_for(
        results,
        expected_text,
        uncertain_note="single-word public media smoke; WER is recorded but not release-gated",
        global_note="Reference text comes from the public real-media fixture manifest.",
    )
    scored = import_llm_reference(results, "SmolLM corrected reference fixture:\n```json\n" + json.dumps(reference) + "\n```")
    scored_path = output_dir / "scored_report.json"
    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    scored_results = dict(results)
    if scored.get("status") == "scored":
        scored_results["reference_scores"] = scored["scores"]
    scored_html = output_dir / "compare_scored.html"
    scored_html.write_text(build_html_report(scored_results), encoding="utf-8", newline="\n")

    dependency_report_failures, dependency_report_details = dependency_resolution_report_failures(results, expected_groups={"whisper_cpp", "llama_cpp"})
    failures: list[str] = list(dependency_report_failures)
    if not transcript.strip():
        failures.append("whisper.cpp public-media transcript was empty")
    for name in ["results.json", "results.txt", "benchmark.csv", "compare.html", "scored_report.json", "compare_scored.html"]:
        if not (output_dir / name).exists():
            failures.append(f"missing report artifact {name}")
    scored_html_text = scored_html.read_text(encoding="utf-8")
    if "Loaded precomputed LLM-corrected reference scores" not in scored_html_text:
        failures.append("compare_scored.html missing precomputed score marker")
    run_id = runs[0]["model"]["candidate_id"] if runs else ""
    score = scored.get("scores", {}).get(run_id, {})
    if scored.get("status") != "scored" or score.get("normalized_wer") is None:
        failures.append("whisper.cpp public-media scored reference was not produced")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "whisper.cpp GGML transcribed real public Wikimedia media, then SmolLM scoring/report validation completed."
            if not failures
            else "whisper.cpp real public media SmolLM grading validation failed."
        ),
        details={
            **fixture_details,
            **scan_details,
            "repo_id": REPO_ID,
            "model_file": MODEL_FILE,
            "model_sha256": sha256(model_path),
            "transcript": transcript,
            "expected_text": expected_text,
            "normalized_wer": normalized_wer,
            "quality_bearing": True,
            "quality_note": "Short public-media WER is recorded but not release-gated.",
            "output_dir": str(output_dir),
            "score_status": scored.get("status"),
            "whisper_cpp_score": {
                "candidate_id": run_id,
                "normalized_wer": score.get("normalized_wer"),
                "balanced_score": score.get("balanced_score"),
                "balanced_rank": score.get("balanced_rank"),
                "alignment_mode": score.get("alignment_mode"),
            },
            "dependency_versions": package_versions(["pywhispercpp", "llama-cpp-python"]),
            **dependency_report_details,
            "failures": failures,
        },
        artifacts=[
            SMOLLM_PATH,
            model_path,
            source,
            output_dir / "results.json",
            output_dir / "results.txt",
            output_dir / "benchmark.csv",
            output_dir / "compare.html",
            scored_path,
            scored_html,
        ],
    )


def _run_speech_quality(row_id: str, evidence_dir: Path, allow_downloads: bool) -> dict:
    model_path_or_row = _ensure_whisper_cpp_fixture(evidence_dir, row_id, allow_downloads, [SMOLLM_PATH])
    if isinstance(model_path_or_row, dict):
        return model_path_or_row
    model_path = model_path_or_row
    config = smoke_config(evidence_dir, "cpu")
    config["runtime"]["llm_context_tokens"] = 1024
    config["runtime"]["llm_reference_max_tokens"] = 128
    config["runtime"]["llm_reference_temperature"] = 0.0
    runnable, unsupported = scan_models(model_path.parent)
    candidates = [candidate for candidate in runnable if candidate.adapter_name == "whisper_cpp"]
    if not candidates:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="whisper.cpp GGML fixture did not scan as a runnable whisper.cpp candidate.",
            details={
                "repo_id": REPO_ID,
                "model_file": MODEL_FILE,
                "runnable": [candidate.adapter_name for candidate in runnable],
                "unsupported": [{"adapter_name": candidate.adapter_name, "warnings": candidate.warnings, "missing": candidate.missing_files} for candidate in unsupported],
            },
            artifacts=[SMOLLM_PATH, model_path],
        )
    llm_candidate, scan_details = _smollm_candidate()
    if llm_candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference/correction LLM candidate.",
            details=scan_details,
            artifacts=[SMOLLM_PATH, model_path],
        )

    source = Path(config["folders"]["input"]) / "whisper_cpp_quality_sapi.wav"
    generate_windows_sapi_wav(source, REFERENCE_TEXT)
    output_dir = process_file_with_candidates(source, [candidates[0]], config, unsupported, reference_llm=llm_candidate)
    if output_dir is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="whisper.cpp speech-quality row did not produce a report directory.",
            details=scan_details,
            artifacts=[SMOLLM_PATH, model_path, source],
        )
    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    runs = results.get("runs", [])
    transcript = "\n".join(chunk.get("text", "") for run in runs for chunk in run.get("transcript_chunks", []))
    normalized_wer = wer(REFERENCE_TEXT, transcript, normalized=True) if transcript.strip() else 1.0
    reference = _reference_for(results, REFERENCE_TEXT)
    scored = import_llm_reference(results, "SmolLM corrected reference fixture:\n```json\n" + json.dumps(reference) + "\n```")
    scored_path = output_dir / "scored_report.json"
    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    scored_results = dict(results)
    if scored.get("status") == "scored":
        scored_results["reference_scores"] = scored["scores"]
    scored_html = output_dir / "compare_scored.html"
    scored_html.write_text(build_html_report(scored_results), encoding="utf-8", newline="\n")

    dependency_report_failures, dependency_report_details = dependency_resolution_report_failures(results, expected_groups={"whisper_cpp", "llama_cpp"})
    failures: list[str] = list(dependency_report_failures)
    if not transcript.strip():
        failures.append("whisper.cpp speech transcript was empty")
    if normalized_wer > 0.85:
        failures.append(f"whisper.cpp normalized WER {normalized_wer:.3f} exceeded threshold 0.850")
    for name in ["results.json", "results.txt", "benchmark.csv", "compare.html", "scored_report.json", "compare_scored.html"]:
        if not (output_dir / name).exists():
            failures.append(f"missing report artifact {name}")
    scored_html_text = scored_html.read_text(encoding="utf-8")
    if "Loaded precomputed LLM-corrected reference scores" not in scored_html_text:
        failures.append("compare_scored.html missing precomputed score marker")
    run_id = runs[0]["model"]["candidate_id"] if runs else ""
    score = scored.get("scores", {}).get(run_id, {})
    if scored.get("status") != "scored" or score.get("normalized_wer") is None:
        failures.append("whisper.cpp scored reference was not produced")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "whisper.cpp GGML transcribed generated speech with acceptable WER, then SmolLM scoring/report validation completed."
            if not failures
            else "whisper.cpp speech-quality SmolLM grading validation failed."
        ),
        details={
            "repo_id": REPO_ID,
            "model_file": MODEL_FILE,
            "reference_text": REFERENCE_TEXT,
            "transcript": transcript,
            "normalized_wer": normalized_wer,
            "max_normalized_wer": 0.85,
            "quality_bearing": True,
            "output_dir": str(output_dir),
            "score_status": scored.get("status"),
            "whisper_cpp_score": {
                "candidate_id": run_id,
                "normalized_wer": score.get("normalized_wer"),
                "balanced_score": score.get("balanced_score"),
                "balanced_rank": score.get("balanced_rank"),
                "alignment_mode": score.get("alignment_mode"),
            },
            "dependency_versions": package_versions(["pywhispercpp", "llama-cpp-python"]),
            **dependency_report_details,
            "failures": failures,
            **scan_details,
        },
        artifacts=[
            SMOLLM_PATH,
            model_path,
            source,
            output_dir / "results.json",
            output_dir / "results.txt",
            output_dir / "benchmark.csv",
            output_dir / "compare.html",
            scored_path,
            scored_html,
        ],
    )


def run(row_id: str, evidence_dir: Path, _install_deps: bool, allow_downloads: bool) -> dict:
    if row_id not in {
        "whisper_cpp_ggml_smollm_grading",
        "whisper_cpp_ggml_speech_smollm_grading",
        "real_public_media_whisper_cpp_ggml_smollm_grading",
        "real_public_video_whisper_cpp_ggml_smollm_grading",
    }:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary=f"Unsupported whisper.cpp SmolLM grading row id: {row_id}",
            details={"row_id": row_id},
        )
    if not SMOLLM_PATH.exists():
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="SmolLM 135M GGUF fixture is not present locally, so whisper.cpp output cannot be graded by the local reference path.",
            block_reason=f"missing {SMOLLM_PATH}",
            external_requirement="download HuggingFaceTB/SmolLM-135M-GGUF Q4_K_M fixture",
        )
    try:
        import llama_cpp  # noqa: F401
    except ModuleNotFoundError:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="llama-cpp-python is not installed, so SmolLM GGUF cannot run after whisper.cpp ASR.",
            block_reason="missing llama_cpp import",
            external_requirement="install llama_cpp dependency group",
            details={"dependency_versions": package_versions(["llama-cpp-python"])},
            artifacts=[SMOLLM_PATH],
        )
    try:
        import pywhispercpp  # noqa: F401
    except ModuleNotFoundError:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="pywhispercpp is not installed, so whisper.cpp plus SmolLM grading cannot run.",
            block_reason="missing pywhispercpp import",
            external_requirement="install whisper_cpp dependency group",
            details={"dependency_versions": package_versions(["pywhispercpp", "llama-cpp-python"])},
            artifacts=[SMOLLM_PATH],
        )
    if row_id == "whisper_cpp_ggml_speech_smollm_grading":
        return _run_speech_quality(row_id, evidence_dir, allow_downloads)
    if row_id in {"real_public_media_whisper_cpp_ggml_smollm_grading", "real_public_video_whisper_cpp_ggml_smollm_grading"}:
        return _run_public_media_quality(row_id, evidence_dir, allow_downloads)

    model_path_or_row = _ensure_whisper_cpp_fixture(evidence_dir, row_id, allow_downloads, [SMOLLM_PATH])
    if isinstance(model_path_or_row, dict):
        return model_path_or_row
    model_path = model_path_or_row

    runnable, unsupported = scan_models(model_path.parent)
    candidates = [candidate for candidate in runnable if candidate.adapter_name == "whisper_cpp"]
    if not candidates:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="whisper.cpp GGML fixture did not scan as a runnable whisper.cpp candidate.",
            details={
                "repo_id": REPO_ID,
                "model_file": MODEL_FILE,
                "runnable": [candidate.adapter_name for candidate in runnable],
                "unsupported": [{"adapter_name": candidate.adapter_name, "warnings": candidate.warnings, "missing": candidate.missing_files} for candidate in unsupported],
            },
            artifacts=[SMOLLM_PATH, model_path],
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
            details={"repo_id": REPO_ID, "model_file": MODEL_FILE, "dependency_versions": package_versions(["pywhispercpp", "llama-cpp-python"])},
            artifacts=[SMOLLM_PATH, model_path],
        )
    finally:
        try:
            adapter.unload()
        except Exception:
            pass
    transcript = result.transcript_chunks[0].text if result.transcript_chunks else ""

    llm_candidate, scan_details = _smollm_candidate()
    if llm_candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference/correction LLM candidate.",
            details=scan_details,
            artifacts=[SMOLLM_PATH, model_path],
        )
    from app.adapters.gguf_llm_reference import GGUFLLMReferenceAdapter

    llm = GGUFLLMReferenceAdapter().load(llm_candidate, {"provider": "cpu", "prefer_gpu": False, "llm_context_tokens": 512})
    response = llm("Answer with the word pass.", max_tokens=8, temperature=0.0)
    generated_text = response["choices"][0]["text"].strip() if isinstance(response, dict) else str(response).strip()
    if not generated_text:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF loaded after whisper.cpp ASR but generated empty text.",
            details={"dependency_versions": package_versions(["pywhispercpp", "llama-cpp-python"])},
            artifacts=[SMOLLM_PATH, model_path],
        )

    source = evidence_dir / "Input" / f"{row_id}.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"easy-asr-bench-whisper-cpp-smollm-grading")
    chunks = [SimpleNamespace(index=0, start_seconds=0.0, end_seconds=1.0, cut_reason="whisper_cpp_fixture", rms_db=-20.0)]
    results = build_results(source, audio_seconds=1.0, chunks=chunks, run_results=[result], unsupported_models=[], media_seconds=0.01)
    results["reference_llm"] = {
        "candidate_id": llm_candidate.candidate_id,
        "display_name": llm_candidate.display_name,
        "path": str(llm_candidate.path),
    }
    results["local_llm_reference_attempt"] = {
        "candidate_id": llm_candidate.candidate_id,
        "display_name": llm_candidate.display_name,
        "status": "generated",
        "raw_response": generated_text,
        "note": "This row proves SmolLM runs after whisper.cpp output. Stable scoring uses the tiny fixture output as structural reference.",
    }
    output_dir = write_all_reports(results, evidence_dir / "Output")
    reference_text = transcript or "[BLANK_AUDIO]"
    reference = _reference_for(results, reference_text)
    scored = import_llm_reference(results, "SmolLM corrected reference fixture:\n```json\n" + json.dumps(reference) + "\n```")
    scored_path = output_dir / "scored_report.json"
    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    scored_results = dict(results)
    if scored.get("status") == "scored":
        scored_results["reference_scores"] = scored["scores"]
    scored_html = output_dir / "compare_scored.html"
    scored_html.write_text(build_html_report(scored_results), encoding="utf-8", newline="\n")

    dependency_report_failures, dependency_report_details = dependency_resolution_report_failures(results, expected_groups={"whisper_cpp", "llama_cpp"})
    failures: list[str] = list(dependency_report_failures)
    for name in ["results.json", "results.txt", "benchmark.csv", "compare.html", "scored_report.json", "compare_scored.html"]:
        if not (output_dir / name).exists():
            failures.append(f"missing report artifact {name}")
    report_text = (output_dir / "results.txt").read_text(encoding="utf-8")
    scored_html_text = scored_html.read_text(encoding="utf-8")
    if "Local GGUF Reference/Correction LLM" not in report_text:
        failures.append("results.txt missing local GGUF reference/correction LLM section")
    if "Loaded precomputed LLM-corrected reference scores" not in scored_html_text:
        failures.append("compare_scored.html missing precomputed score marker")
    run_id = results["runs"][0]["model"]["candidate_id"]
    score = scored.get("scores", {}).get(run_id, {})
    if scored.get("status") != "scored" or score.get("normalized_wer") is None:
        failures.append("whisper.cpp scored reference was not produced")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "whisper.cpp GGML output was followed by SmolLM GGUF generation and scored report validation."
            if not failures
            else "whisper.cpp SmolLM grading validation failed."
        ),
        details={
            "repo_id": REPO_ID,
            "model_file": MODEL_FILE,
            "transcript": transcript,
            "quality_bearing": False,
            "quality_note": "Tiny whisper.cpp blank-audio fixture is used for structural backend/report regression only; real WER proof remains pending.",
            "metrics": result.metrics,
            "errors": result.errors,
            "output_dir": str(output_dir),
            "candidate_id": llm_candidate.candidate_id,
            "generated_text": generated_text,
            "score_status": scored.get("status"),
            "whisper_cpp_score": {
                "candidate_id": run_id,
                "normalized_wer": score.get("normalized_wer"),
                "balanced_score": score.get("balanced_score"),
                "balanced_rank": score.get("balanced_rank"),
                "alignment_mode": score.get("alignment_mode"),
            },
            "dependency_versions": package_versions(["pywhispercpp", "llama-cpp-python"]),
            **dependency_report_details,
            "failures": failures,
            **scan_details,
        },
        artifacts=[
            SMOLLM_PATH,
            model_path,
            output_dir / "results.json",
            output_dir / "results.txt",
            output_dir / "benchmark.csv",
            output_dir / "compare.html",
            scored_path,
            scored_html,
        ],
    )
