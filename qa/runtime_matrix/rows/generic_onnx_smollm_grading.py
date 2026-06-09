from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import numpy as np

from app.adapters.generic_onnx_manifest import GenericOnnxManifestAdapter
from app.dependency_manager import cuda_diagnostics
from app.dependency_manager import install_group_for_config, missing_modules_for_config, recovery_command_for_config
from app.html_report_builder import build_html_report
from app.main import process_file_with_candidates
from app.model_scanner import scan_models
from app.reference_import import import_llm_reference
from app.results_writer import build_results, write_all_reports
from app.scoring import wer
from qa.run_real_tiny_model_smoke import REFERENCE_TEXT, generate_windows_sapi_wav, smoke_config
from qa.runtime_matrix.common import dependency_resolution_report_failures, package_versions, write_row
from qa.runtime_matrix.rows.generic_onnx_ctc_tiny import _provider_for_row, _write_tiny_ctc_fixture
from qa.runtime_matrix.rows.real_public_media_faster_whisper_smollm import _download_fixture
from qa.runtime_matrix.rows.smollm_reference_grading_report import SMOLLM_PATH, _smollm_candidate


GENERIC_ONNX_QUALITY_REPO = "onnx-community/wav2vec2-base-960h-ONNX"
GENERIC_ONNX_QUALITY_MODEL_FILE = "onnx/model_int8.onnx"
GENERIC_ONNX_QUALITY_VOCAB_FILE = "vocab.json"
GENERIC_ONNX_QUALITY_MODEL_URL = (
    "https://huggingface.co/"
    + GENERIC_ONNX_QUALITY_REPO
    + "/resolve/main/"
    + GENERIC_ONNX_QUALITY_MODEL_FILE
)
GENERIC_ONNX_QUALITY_VOCAB_URL = (
    "https://huggingface.co/"
    + GENERIC_ONNX_QUALITY_REPO
    + "/resolve/main/"
    + GENERIC_ONNX_QUALITY_VOCAB_FILE
)


def _reference_for_onnx(results: dict) -> dict:
    return {
        "schema": "easy_asr_bench.llm_reference.v1",
        "source_sha256": results["source"]["sha256"],
        "reference_type": "llm_corrected_reference",
        "segments": [
            {
                "chunk_id": chunk["chunk_id"],
                "start_seconds": chunk["start_seconds"],
                "end_seconds": chunk["end_seconds"],
                "text": "ab",
                "uncertain": [],
            }
            for chunk in results.get("chunk_plan", {}).get("chunks", [])
        ],
        "global_notes": ["Reference text comes from the deterministic generated ONNX CTC fixture."],
    }


def _block_if_provider_unavailable(row_id: str, provider: str) -> dict | None:
    diagnostics = cuda_diagnostics()
    providers = diagnostics.get("onnxruntime_providers", [])
    if provider == "directml":
        if "DmlExecutionProvider" not in providers:
            return {
                "summary": "ONNX DirectML plus SmolLM grading requires DmlExecutionProvider, which is not visible.",
                "block_reason": "DmlExecutionProvider missing from onnxruntime providers",
                "external_requirement": "Windows DirectML-capable GPU with onnxruntime-directml installed and provider visible",
                "details": {"cuda_provider_checks": diagnostics, "dependency_versions": package_versions(["onnxruntime", "onnxruntime-directml"])},
            }
    return None


def _ensure_onnx_deps(row_id: str, evidence_dir: Path, provider: str, install_deps: bool, artifacts: list[Path]) -> dict | None:
    config = smoke_config(evidence_dir, provider)
    missing = missing_modules_for_config("onnx", config)
    if not missing:
        return None
    repair_log = evidence_dir / "onnx_repair.log"
    repair_command = recovery_command_for_config("onnx", config)
    if install_deps:
        try:
            install_group_for_config("onnx", Path.cwd(), config, log_path=repair_log)
        except Exception as exc:
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary="ONNX dependency repair failed before Generic ONNX quality validation.",
                block_reason=f"{type(exc).__name__}: {exc}",
                external_requirement=repair_command,
                details={"missing_before": missing, "dependency_versions": package_versions(["onnx", "onnxruntime", "onnxruntime-directml"])},
                artifacts=[*artifacts, repair_log],
            )
        missing = missing_modules_for_config("onnx", config)
    if missing:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="ONNX dependency group is not currently runnable.",
            block_reason=", ".join(missing),
            external_requirement=repair_command,
            details={"missing": missing, "dependency_versions": package_versions(["onnx", "onnxruntime", "onnxruntime-directml"])},
            artifacts=[*artifacts, repair_log],
        )
    return None


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Easy-ASR-Bench-runtime-matrix"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _write_public_ctc_manifest(model_dir: Path) -> Path:
    manifest = {
        "schema": "easy_asr_bench.model_manifest.v1",
        "display_name": "wav2vec2 base 960h ONNX int8 CTC",
        "task": "automatic-speech-recognition",
        "precision": "int8",
        "files": {"model": "model.onnx"},
        "inputs": {"waveform": {"name": "input_values"}},
        "outputs": {"logits": "logits"},
        "preprocessing": {"type": "raw_waveform", "normalize": True},
        "decoding": {"type": "ctc", "blank_token_id": 0, "vocab_file": "vocab.json"},
    }
    path = model_dir / "modelbench.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def _find_cached_public_ctc_fixture() -> Path | None:
    folder_name = GENERIC_ONNX_QUALITY_REPO.replace("/", "__")
    for root_name in ("Temp", "Models", "Cache"):
        root = Path.cwd() / root_name
        if not root.exists():
            continue
        for candidate in root.rglob(folder_name):
            if not candidate.is_dir():
                continue
            if (candidate / "model.onnx").exists() and (candidate / "vocab.json").exists():
                return candidate
    return None


def _copy_cached_public_ctc_fixture(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    ignore = shutil.ignore_patterns(".hf_cache", "__pycache__")
    shutil.copytree(source, destination, ignore=ignore)
    _write_public_ctc_manifest(destination)


def _ensure_public_ctc_fixture(row_id: str, evidence_dir: Path, allow_downloads: bool, artifacts: list[Path]) -> dict | Path:
    model_dir = evidence_dir / "Models" / GENERIC_ONNX_QUALITY_REPO.replace("/", "__")
    model_path = model_dir / "model.onnx"
    vocab_path = model_dir / "vocab.json"
    manifest_path = model_dir / "modelbench.json"
    if model_path.exists() and vocab_path.exists():
        _write_public_ctc_manifest(model_dir)
        return model_dir
    cached = _find_cached_public_ctc_fixture()
    if cached is not None:
        _copy_cached_public_ctc_fixture(cached, model_dir)
        return model_dir
    if not allow_downloads:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary=f"{GENERIC_ONNX_QUALITY_REPO} int8 ONNX CTC fixture is not cached locally.",
            block_reason=f"missing local fixture {model_dir}",
            external_requirement=f"rerun with --allow-downloads to download {GENERIC_ONNX_QUALITY_REPO} {GENERIC_ONNX_QUALITY_MODEL_FILE}",
            details={
                "repo_id": GENERIC_ONNX_QUALITY_REPO,
                "model_file": GENERIC_ONNX_QUALITY_MODEL_FILE,
                "vocab_file": GENERIC_ONNX_QUALITY_VOCAB_FILE,
            },
            artifacts=artifacts,
        )
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        _download(GENERIC_ONNX_QUALITY_MODEL_URL, model_path)
        _download(GENERIC_ONNX_QUALITY_VOCAB_URL, vocab_path)
        _write_public_ctc_manifest(model_dir)
    except Exception as exc:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary=f"Could not download {GENERIC_ONNX_QUALITY_REPO} int8 ONNX CTC fixture.",
            block_reason=f"{type(exc).__name__}: {exc}",
            external_requirement=f"network access to https://huggingface.co/{GENERIC_ONNX_QUALITY_REPO}",
            details={
                "repo_id": GENERIC_ONNX_QUALITY_REPO,
                "model_url": GENERIC_ONNX_QUALITY_MODEL_URL,
                "vocab_url": GENERIC_ONNX_QUALITY_VOCAB_URL,
            },
            artifacts=[*artifacts, model_path, vocab_path, manifest_path],
        )
    return model_dir


def _reference_for_text(results: dict, text: str) -> dict:
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
                "uncertain": [],
            }
            for chunk in results.get("chunk_plan", {}).get("chunks", [])
        ],
        "global_notes": ["Reference text is the known Windows SAPI smoke phrase used for quality-bearing runtime validation."],
    }


def _reference_for_public_media(results: dict, text: str) -> dict:
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
                "uncertain": ["single-word public media smoke; WER is recorded but not release-gated"],
            }
            for chunk in results.get("chunk_plan", {}).get("chunks", [])
        ],
        "global_notes": ["Reference text comes from the public real-media fixture manifest."],
    }


def _run_public_quality(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    provider = "cpu"
    dependency_block = _ensure_onnx_deps(row_id, evidence_dir, provider, install_deps, [SMOLLM_PATH])
    if dependency_block is not None:
        return dependency_block
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ModuleNotFoundError as exc:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="ONNX dependency group is not installed, so public Generic ONNX CTC quality validation cannot run.",
            block_reason=f"missing {exc.name}",
            external_requirement=recovery_command_for_config("onnx", smoke_config(evidence_dir, provider)),
            details={"dependency_versions": package_versions(["onnx", "onnxruntime", "onnxruntime-directml"])},
            artifacts=[SMOLLM_PATH],
        )
    fixture_or_row = _ensure_public_ctc_fixture(row_id, evidence_dir, allow_downloads, [SMOLLM_PATH])
    if isinstance(fixture_or_row, dict):
        return fixture_or_row
    model_dir = fixture_or_row
    config = smoke_config(evidence_dir, provider)
    config["runtime"]["cpu_threads"] = 1
    config["runtime"]["max_chunk_seconds"] = 5
    config["runtime"]["chunk_stride_seconds"] = 0
    config["runtime"]["llm_context_tokens"] = 1024
    config["runtime"]["llm_reference_max_tokens"] = 128
    config["runtime"]["llm_reference_temperature"] = 0.0

    runnable, unsupported = scan_models(model_dir)
    candidates = [candidate for candidate in runnable if candidate.adapter_name == "generic_onnx_manifest"]
    if not candidates:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary=f"{GENERIC_ONNX_QUALITY_REPO} did not scan as a runnable Generic ONNX CTC manifest model.",
            details={
                "repo_id": GENERIC_ONNX_QUALITY_REPO,
                "runnable": [candidate.adapter_name for candidate in runnable],
                "unsupported": [{"adapter_name": candidate.adapter_name, "missing": candidate.missing_files, "warnings": candidate.warnings} for candidate in unsupported],
            },
            artifacts=[SMOLLM_PATH, model_dir / "model.onnx", model_dir / "vocab.json", model_dir / "modelbench.json"],
        )
    llm_candidate, scan_details = _smollm_candidate()
    if llm_candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference/correction LLM candidate.",
            details=scan_details,
            artifacts=[SMOLLM_PATH, model_dir / "model.onnx", model_dir / "vocab.json", model_dir / "modelbench.json"],
        )

    source = Path(config["folders"]["input"]) / "generic_onnx_ctc_quality_sapi.wav"
    generate_windows_sapi_wav(source, REFERENCE_TEXT)
    output_dir = process_file_with_candidates(source, [candidates[0]], config, unsupported, reference_llm=llm_candidate)
    if output_dir is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Generic ONNX CTC quality row did not produce a report directory.",
            details={"repo_id": GENERIC_ONNX_QUALITY_REPO, **scan_details},
            artifacts=[SMOLLM_PATH, model_dir / "model.onnx", model_dir / "vocab.json", model_dir / "modelbench.json", source],
        )
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    runs = results.get("runs", [])
    transcript = "\n".join(chunk.get("text", "") for run in runs for chunk in run.get("transcript_chunks", []))
    normalized_wer = wer(REFERENCE_TEXT, transcript, normalized=True) if transcript.strip() else 1.0
    reference = _reference_for_text(results, REFERENCE_TEXT)
    scored = import_llm_reference(results, "SmolLM corrected reference fixture:\n```json\n" + json.dumps(reference) + "\n```")
    scored_path = output_dir / "scored_report.json"
    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    scored_html = output_dir / "compare_scored.html"
    scored_html.write_text(build_html_report(scored), encoding="utf-8", newline="\n")

    dependency_report_failures, dependency_report_details = dependency_resolution_report_failures(results, expected_groups={"onnx", "llama_cpp"})
    failures: list[str] = list(dependency_report_failures)
    if not transcript.strip():
        failures.append("Generic ONNX CTC speech transcript was empty")
    if normalized_wer > 0.85:
        failures.append(f"Generic ONNX CTC normalized WER {normalized_wer:.3f} exceeded threshold 0.850")
    for name in ["results.json", "results.txt", "benchmark.csv", "compare.html", "scored_report.json", "compare_scored.html"]:
        if not (output_dir / name).exists():
            failures.append(f"missing report artifact {name}")
    report_text = (output_dir / "results.txt").read_text(encoding="utf-8")
    scored_html_text = scored_html.read_text(encoding="utf-8")
    if "Local GGUF Reference/Correction LLM" not in report_text:
        failures.append("results.txt missing local GGUF reference/correction LLM section")
    if "Loaded precomputed LLM-corrected reference scores" not in scored_html_text:
        failures.append("compare_scored.html missing precomputed score marker")
    run_id = runs[0]["model"]["candidate_id"] if runs else ""
    score = scored.get("scores", {}).get(run_id, {})
    if scored.get("status") != "scored" or score.get("normalized_wer") is None:
        failures.append("Generic ONNX CTC scored reference was not produced")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "Public int8 Generic ONNX CTC model transcribed generated speech with acceptable WER, then SmolLM scoring/report validation completed."
            if not failures
            else "Public Generic ONNX CTC quality SmolLM grading validation failed."
        ),
        details={
            "repo_id": GENERIC_ONNX_QUALITY_REPO,
            "model_file": GENERIC_ONNX_QUALITY_MODEL_FILE,
            "adapter_name": "generic_onnx_manifest",
            "reference_text": REFERENCE_TEXT,
            "transcript": transcript,
            "normalized_wer": normalized_wer,
            "max_normalized_wer": 0.85,
            "quality_bearing": True,
            "output_dir": str(output_dir),
            "score_status": scored.get("status"),
            "generic_onnx_ctc_score": {
                "candidate_id": run_id,
                "normalized_wer": score.get("normalized_wer"),
                "balanced_score": score.get("balanced_score"),
                "balanced_rank": score.get("balanced_rank"),
                "alignment_mode": score.get("alignment_mode"),
            },
            "dependency_versions": package_versions(["onnx", "onnxruntime", "onnxruntime-directml", "llama-cpp-python"]),
            **dependency_report_details,
            "rejected_smaller_candidate": {
                "repo_id": GENERIC_ONNX_QUALITY_REPO,
                "model_file": "onnx/model_q4f16.onnx",
                "observed_failure": "ONNX Runtime CPU preflight failed during graph initialization on MatMulNBits/SimplifiedLayerNormFusion; int8 is the smallest probed public CTC candidate that completed this row.",
            },
            "failures": failures,
            **scan_details,
        },
        artifacts=[
            SMOLLM_PATH,
            model_dir / "model.onnx",
            model_dir / "vocab.json",
            model_dir / "modelbench.json",
            source,
            output_dir / "results.json",
            output_dir / "results.txt",
            output_dir / "benchmark.csv",
            output_dir / "compare.html",
            scored_path,
            scored_html,
        ],
    )


def _run_real_public_media_quality(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    provider = "cpu"
    dependency_block = _ensure_onnx_deps(row_id, evidence_dir, provider, install_deps, [SMOLLM_PATH])
    if dependency_block is not None:
        return dependency_block
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ModuleNotFoundError as exc:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="ONNX dependency group is not installed, so real public Generic ONNX CTC validation cannot run.",
            block_reason=f"missing {exc.name}",
            external_requirement=recovery_command_for_config("onnx", smoke_config(evidence_dir, provider)),
            details={"dependency_versions": package_versions(["onnx", "onnxruntime", "onnxruntime-directml"])},
            artifacts=[SMOLLM_PATH],
        )
    fixture_or_row = _ensure_public_ctc_fixture(row_id, evidence_dir, allow_downloads, [SMOLLM_PATH])
    if isinstance(fixture_or_row, dict):
        return fixture_or_row
    model_dir = fixture_or_row
    fixture_id = "wikimedia_public_domain_spoken_words_webm" if row_id == "real_public_video_generic_onnx_ctc_smollm_grading_cpu" else "wikimedia_cc0_word_wav"
    source, fixture_details, fixture_error = _download_fixture(fixture_id, evidence_dir, allow_downloads)
    if fixture_error or source is None:
        return write_row(
            row_id,
            "blocked" if not allow_downloads else "fail",
            evidence_dir,
            summary="Real public media fixture is not available for Generic ONNX CTC ASR+SmolLM validation.",
            block_reason=fixture_error if not allow_downloads else None,
            external_requirement="rerun with --allow-downloads after source/license review" if not allow_downloads else None,
            details={"repo_id": GENERIC_ONNX_QUALITY_REPO, **fixture_details},
            artifacts=[SMOLLM_PATH, model_dir / "model.onnx", model_dir / "vocab.json", model_dir / "modelbench.json"],
        )
    config = smoke_config(evidence_dir, provider)
    config["runtime"]["cpu_threads"] = 1
    config["runtime"]["max_chunk_seconds"] = 5
    config["runtime"]["chunk_stride_seconds"] = 0
    config["runtime"]["llm_context_tokens"] = 1024
    config["runtime"]["llm_reference_max_tokens"] = 128
    config["runtime"]["llm_reference_temperature"] = 0.0

    runnable, unsupported = scan_models(model_dir)
    candidates = [candidate for candidate in runnable if candidate.adapter_name == "generic_onnx_manifest"]
    if not candidates:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary=f"{GENERIC_ONNX_QUALITY_REPO} did not scan as a runnable Generic ONNX CTC manifest model.",
            details={
                "repo_id": GENERIC_ONNX_QUALITY_REPO,
                **fixture_details,
                "runnable": [candidate.adapter_name for candidate in runnable],
                "unsupported": [{"adapter_name": candidate.adapter_name, "missing": candidate.missing_files, "warnings": candidate.warnings} for candidate in unsupported],
            },
            artifacts=[SMOLLM_PATH, model_dir / "model.onnx", model_dir / "vocab.json", model_dir / "modelbench.json", source],
        )
    llm_candidate, scan_details = _smollm_candidate()
    if llm_candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference/correction LLM candidate.",
            details={"repo_id": GENERIC_ONNX_QUALITY_REPO, **fixture_details, "smollm_scan": scan_details},
            artifacts=[SMOLLM_PATH, model_dir / "model.onnx", model_dir / "vocab.json", model_dir / "modelbench.json", source],
        )

    output_dir = process_file_with_candidates(source, [candidates[0]], config, unsupported, reference_llm=llm_candidate)
    if output_dir is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Generic ONNX CTC real public-media row did not produce a report directory.",
            details={"repo_id": GENERIC_ONNX_QUALITY_REPO, **fixture_details, "smollm_scan": scan_details},
            artifacts=[SMOLLM_PATH, model_dir / "model.onnx", model_dir / "vocab.json", model_dir / "modelbench.json", source],
        )
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    runs = results.get("runs", [])
    transcript = "\n".join(chunk.get("text", "") for run in runs for chunk in run.get("transcript_chunks", []))
    expected_text = fixture_details["fixture"].get("expected_text") or ""
    normalized_wer = wer(expected_text, transcript, normalized=True) if expected_text and transcript.strip() else 1.0
    reference = _reference_for_public_media(results, expected_text)
    scored = import_llm_reference(results, "SmolLM corrected reference fixture:\n```json\n" + json.dumps(reference) + "\n```")
    scored_path = output_dir / "scored_report.json"
    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    scored_html = output_dir / "compare_scored.html"
    scored_html.write_text(build_html_report(scored), encoding="utf-8", newline="\n")

    dependency_report_failures, dependency_report_details = dependency_resolution_report_failures(results, expected_groups={"onnx", "llama_cpp"})
    failures: list[str] = list(dependency_report_failures)
    if not transcript.strip():
        failures.append("Generic ONNX CTC public-media transcript was empty")
    for name in ["results.json", "results.txt", "benchmark.csv", "compare.html", "scored_report.json", "compare_scored.html"]:
        if not (output_dir / name).exists():
            failures.append(f"missing report artifact {name}")
    scored_html_text = scored_html.read_text(encoding="utf-8")
    if "Loaded precomputed LLM-corrected reference scores" not in scored_html_text:
        failures.append("compare_scored.html missing precomputed score marker")
    run_id = runs[0]["model"]["candidate_id"] if runs else ""
    score = scored.get("scores", {}).get(run_id, {})
    if scored.get("status") != "scored" or score.get("normalized_wer") is None:
        failures.append("Generic ONNX CTC public-media scored reference was not produced")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "Generic ONNX CTC transcribed real public Wikimedia media, then SmolLM scoring/report validation completed."
            if not failures
            else "Generic ONNX CTC real public-media SmolLM grading validation failed."
        ),
        details={
            "repo_id": GENERIC_ONNX_QUALITY_REPO,
            "model_file": GENERIC_ONNX_QUALITY_MODEL_FILE,
            "adapter_name": "generic_onnx_manifest",
            **fixture_details,
            "generic_onnx_runnable_count": len(candidates),
            "generic_onnx_unsupported_count": len(unsupported),
            "smollm_scan": scan_details,
            "transcript": transcript,
            "expected_text": expected_text,
            "normalized_wer": normalized_wer,
            "quality_bearing": True,
            "quality_note": "Short public-media WER is recorded but not release-gated.",
            "output_dir": str(output_dir),
            "score_status": scored.get("status"),
            "generic_onnx_ctc_score": {
                "candidate_id": run_id,
                "normalized_wer": score.get("normalized_wer"),
                "balanced_score": score.get("balanced_score"),
                "balanced_rank": score.get("balanced_rank"),
                "alignment_mode": score.get("alignment_mode"),
            },
            "dependency_versions": package_versions(["onnx", "onnxruntime", "onnxruntime-directml", "llama-cpp-python"]),
            **dependency_report_details,
            "rejected_smaller_candidate": {
                "repo_id": GENERIC_ONNX_QUALITY_REPO,
                "model_file": "onnx/model_q4f16.onnx",
                "observed_failure": "ONNX Runtime CPU preflight failed during graph initialization on MatMulNBits/SimplifiedLayerNormFusion; int8 is the smallest probed public CTC candidate that completed this row.",
            },
            "failures": failures,
        },
        artifacts=[
            SMOLLM_PATH,
            model_dir / "model.onnx",
            model_dir / "vocab.json",
            model_dir / "modelbench.json",
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
    if row_id not in {
        "generic_onnx_smollm_grading_cpu",
        "generic_onnx_smollm_grading_directml",
        "generic_onnx_ctc_quality_smollm_grading_cpu",
        "real_public_media_generic_onnx_ctc_smollm_grading_cpu",
        "real_public_video_generic_onnx_ctc_smollm_grading_cpu",
    }:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary=f"Unsupported Generic ONNX SmolLM grading row id: {row_id}",
            details={"row_id": row_id},
        )
    if not SMOLLM_PATH.exists():
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="SmolLM 135M GGUF fixture is not present locally, so Generic ONNX output cannot be graded by the local reference path.",
            block_reason=f"missing {SMOLLM_PATH}",
            external_requirement="download HuggingFaceTB/SmolLM-135M-GGUF Q4_K_M fixture",
        )
    try:
        from app.dependency_manager import prepare_llama_cpp_dll_search_path

        prepare_llama_cpp_dll_search_path()
        import llama_cpp  # noqa: F401
    except ModuleNotFoundError:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="llama-cpp-python is not installed, so SmolLM GGUF cannot run after Generic ONNX ASR.",
            block_reason="missing llama_cpp import",
            external_requirement="install llama_cpp dependency group",
            details={"dependency_versions": package_versions(["llama-cpp-python"])},
            artifacts=[SMOLLM_PATH],
        )
    if row_id == "generic_onnx_ctc_quality_smollm_grading_cpu":
        return _run_public_quality(row_id, evidence_dir, _install_deps, _allow_downloads)
    if row_id in {"real_public_media_generic_onnx_ctc_smollm_grading_cpu", "real_public_video_generic_onnx_ctc_smollm_grading_cpu"}:
        return _run_real_public_media_quality(row_id, evidence_dir, _install_deps, _allow_downloads)
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ModuleNotFoundError as exc:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="ONNX dependency group is not installed, so Generic ONNX plus SmolLM grading cannot run.",
            block_reason=f"missing {exc.name}",
            external_requirement="python -m pip install -r requirements/onnx.txt",
            details={"dependency_versions": package_versions(["onnx", "onnxruntime", "onnxruntime-directml"])},
            artifacts=[SMOLLM_PATH],
        )

    provider = _provider_for_row(row_id)
    provider_block = _block_if_provider_unavailable(row_id, provider)
    if provider_block is not None:
        return write_row(row_id, "blocked", evidence_dir, artifacts=[SMOLLM_PATH], **provider_block)

    model_dir = evidence_dir / "tiny_onnx_ctc"
    model_artifacts = _write_tiny_ctc_fixture(model_dir)
    runnable, unsupported = scan_models(model_dir)
    candidates = [candidate for candidate in runnable if candidate.adapter_name == "generic_onnx_manifest"]
    if not candidates:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Tiny ONNX CTC manifest fixture was not discovered as a runnable Generic ONNX CTC model.",
            details={"runnable_count": len(runnable), "unsupported_count": len(unsupported)},
            artifacts=[SMOLLM_PATH, *model_artifacts],
        )

    adapter = GenericOnnxManifestAdapter()
    candidate = candidates[0]
    try:
        adapter.load(candidate, {"provider": provider, "prefer_gpu": provider != "cpu", "cpu_threads": 1})
        result = adapter.transcribe_chunks(
            [SimpleNamespace(samples=np.zeros(1600, dtype=np.float32))],
            [{"chunk_id": "0001", "start_seconds": 0.0, "end_seconds": 0.1}],
        )
    except Exception as exc:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary=f"Tiny Generic ONNX CTC fixture could not run with requested provider {provider}.",
            block_reason=f"{type(exc).__name__}: {exc}",
            external_requirement="repair ONNX Runtime provider package or rerun with CPU provider",
            details={"provider": provider, "dependency_versions": package_versions(["onnx", "onnxruntime", "onnxruntime-directml", "onnxruntime-openvino", "onnxruntime-gpu"])},
            artifacts=[SMOLLM_PATH, *model_artifacts],
        )
    transcript = result.transcript_chunks[0].text if result.transcript_chunks else ""
    if transcript != "ab":
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Tiny Generic ONNX CTC fixture ran but decoded the wrong transcript.",
            details={"provider": provider, "transcript": transcript, "metrics": result.metrics},
            artifacts=[SMOLLM_PATH, *model_artifacts],
        )

    llm_candidate, scan_details = _smollm_candidate()
    if llm_candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference/correction LLM candidate.",
            details=scan_details,
            artifacts=[SMOLLM_PATH, *model_artifacts],
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
            summary="SmolLM GGUF loaded after Generic ONNX ASR but generated empty text.",
            details={"dependency_versions": package_versions(["llama-cpp-python"])},
            artifacts=[SMOLLM_PATH, *model_artifacts],
        )

    source = evidence_dir / "Input" / f"{row_id}.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"easy-asr-bench-generic-onnx-smollm-grading")
    chunks = [SimpleNamespace(index=0, start_seconds=0.0, end_seconds=0.1, cut_reason="onnx_fixture", rms_db=-20.0)]
    results = build_results(source, audio_seconds=0.1, chunks=chunks, run_results=[result], unsupported_models=[], media_seconds=0.01)
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
        "note": "This row proves SmolLM runs after Generic ONNX CTC output. Stable scoring uses the deterministic generated ONNX reference text.",
    }
    output_dir = write_all_reports(results, evidence_dir / "Output")
    reference = _reference_for_onnx(results)
    scored = import_llm_reference(results, "SmolLM corrected reference fixture:\n```json\n" + json.dumps(reference) + "\n```")
    scored_path = output_dir / "scored_report.json"
    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    scored_html = output_dir / "compare_scored.html"
    scored_html.write_text(build_html_report(scored), encoding="utf-8", newline="\n")

    dependency_report_failures, dependency_report_details = dependency_resolution_report_failures(results, expected_groups={"onnx", "llama_cpp"})
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
    if provider == "directml" and "DmlExecutionProvider" not in scored_html_text:
        failures.append("compare_scored.html missing DmlExecutionProvider marker")
    run_id = results["runs"][0]["model"]["candidate_id"]
    score = scored.get("scores", {}).get(run_id, {})
    if scored.get("status") != "scored" or score.get("normalized_wer") != 0:
        failures.append("Generic ONNX scored reference did not produce zero WER")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            f"Generic ONNX CTC output with provider {provider} was followed by SmolLM GGUF generation and scored report validation."
            if not failures
            else "Generic ONNX SmolLM grading validation failed."
        ),
        details={
            "provider": provider,
            "transcript": transcript,
            "metrics": result.metrics,
            "output_dir": str(output_dir),
            "candidate_id": llm_candidate.candidate_id,
            "generated_text": generated_text,
            "score_status": scored.get("status"),
            "generic_onnx_score": {
                "candidate_id": run_id,
                "normalized_wer": score.get("normalized_wer"),
                "balanced_score": score.get("balanced_score"),
                "balanced_rank": score.get("balanced_rank"),
                "alignment_mode": score.get("alignment_mode"),
            },
            "dependency_versions": package_versions(["onnx", "onnxruntime", "onnxruntime-directml", "onnxruntime-openvino", "onnxruntime-gpu", "llama-cpp-python"]),
            **dependency_report_details,
            "failures": failures,
            **scan_details,
        },
        artifacts=[
            SMOLLM_PATH,
            *model_artifacts,
            output_dir / "results.json",
            output_dir / "results.txt",
            output_dir / "benchmark.csv",
            output_dir / "compare.html",
            scored_path,
            scored_html,
        ],
    )
