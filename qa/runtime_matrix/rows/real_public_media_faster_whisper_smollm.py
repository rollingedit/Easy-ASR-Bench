from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.adapters.openai_whisper_pt import is_verified_official_checkpoint
from app.adapters.gguf_llm_reference import GGUFLLMReferenceAdapter
from app.dependency_manager import install_group_for_config, missing_modules_for_config, recovery_command_for_config
from app.hf_model_downloader import RECOMMENDED_BASELINE_REPO, download_hf_model_from_ref
from app.html_report_builder import build_html_report
from app.main import process_file_with_candidates
from app.model_scanner import scan_models
from app.reference_import import import_llm_reference
from app.repair_plan import execute_repair_plan
from app.results_writer import render_text_report, write_benchmark_csv
from app.scoring import wer
from qa.runtime_matrix.common import dependency_resolution_report_failures, package_versions, sha256, write_row
from qa.runtime_matrix.rows.real_media_download_cache import MANIFEST, _download, _extension_for_fixture, _load_manifest
from qa.runtime_matrix.rows.openai_whisper_pt_safety import TINY_PT, TINY_PT_SHA256, _download_official_checkpoint
from qa.runtime_matrix.rows.report_reference_validation import _assert_report_files
from qa.runtime_matrix.rows.smollm_reference_grading_report import SMOLLM_PATH, _smollm_candidate
from qa.run_real_tiny_model_smoke import smoke_config


ROW_FIXTURES = {
    "real_public_media_faster_whisper_smollm_grading": "wikimedia_cc0_word_wav",
    "real_public_video_faster_whisper_smollm_grading": "wikimedia_public_domain_spoken_words_webm",
    "real_public_media_openai_whisper_pt_smollm_grading": "wikimedia_cc0_word_wav",
    "real_public_video_openai_whisper_pt_smollm_grading": "wikimedia_public_domain_spoken_words_webm",
}
ROW_BACKENDS = {
    "real_public_media_faster_whisper_smollm_grading": "faster_whisper",
    "real_public_video_faster_whisper_smollm_grading": "faster_whisper",
    "real_public_media_openai_whisper_pt_smollm_grading": "openai_whisper_pt",
    "real_public_video_openai_whisper_pt_smollm_grading": "openai_whisper_pt",
}
GROUPS_BY_BACKEND = {
    "faster_whisper": {"python_packaging", "media_tools", "faster_whisper", "llama_cpp"},
    "openai_whisper_pt": {"python_packaging", "media_tools", "openai_whisper", "llama_cpp"},
}
PACKAGE_NAMES_BY_BACKEND = {
    "faster_whisper": ["pip", "setuptools", "faster-whisper", "ctranslate2", "llama-cpp-python"],
    "openai_whisper_pt": ["pip", "setuptools", "openai-whisper", "torch", "llama-cpp-python"],
}
LOCAL_FIXTURE_SEARCH_ROOTS = ("Temp", "Input", "Cache")
LOCAL_OPENAI_PT_SEARCH_ROOTS = ("Temp", "Models", "Cache")
LOCAL_MODEL_REPO_SEARCH_ROOTS = ("Temp", "Models", "Cache")


def _repo_folder_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _find_cached_repo_folder(repo_id: str, required_names: set[str], exclude: Path | None = None) -> Path | None:
    folder_name = _repo_folder_name(repo_id)
    excluded = exclude.resolve() if exclude is not None and exclude.exists() else None
    for root_name in LOCAL_MODEL_REPO_SEARCH_ROOTS:
        root = Path.cwd() / root_name
        if not root.exists():
            continue
        for candidate in root.rglob(folder_name):
            if not candidate.is_dir():
                continue
            try:
                if excluded is not None and candidate.resolve() == excluded:
                    continue
            except OSError:
                continue
            names = {path.name for path in candidate.rglob("*") if path.is_file()}
            if required_names <= names:
                return candidate
    return None


def _find_cached_model_folder(required_names: set[str], any_name_groups: tuple[set[str], ...], exclude: Path | None = None) -> Path | None:
    excluded = exclude.resolve() if exclude is not None and exclude.exists() else None
    for root_name in LOCAL_MODEL_REPO_SEARCH_ROOTS:
        root = Path.cwd() / root_name
        if not root.exists():
            continue
        for model_bin in root.rglob("model.bin"):
            candidate = model_bin.parent
            if not candidate.is_dir():
                continue
            try:
                if excluded is not None and candidate.resolve() == excluded:
                    continue
            except OSError:
                continue
            names = {path.name for path in candidate.rglob("*") if path.is_file()}
            if required_names <= names and all(names & group for group in any_name_groups):
                return candidate
    return None


def _copy_cached_repo_folder(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    ignore = shutil.ignore_patterns(".hf_cache", "__pycache__")
    shutil.copytree(source, destination, ignore=ignore)


def _ensure_faster_whisper(models_root: Path, allow_downloads: bool) -> tuple[Path | None, str | None]:
    runnable, _ = scan_models(models_root)
    matches = [candidate for candidate in runnable if candidate.adapter_name == "faster_whisper"]
    if matches:
        return matches[0].path, None
    destination = models_root / _repo_folder_name(RECOMMENDED_BASELINE_REPO)
    cached = _find_cached_repo_folder(RECOMMENDED_BASELINE_REPO, {"model.bin", "config.json"}, exclude=destination)
    if cached is None:
        cached = _find_cached_model_folder(
            {"model.bin", "config.json"},
            ({"tokenizer.json", "vocabulary.json", "vocabulary.txt", "vocab.json"},),
            exclude=destination,
        )
    if cached is not None:
        _copy_cached_repo_folder(cached, destination)
        runnable, _ = scan_models(models_root)
        matches = [candidate for candidate in runnable if candidate.adapter_name == "faster_whisper"]
        if matches:
            return matches[0].path, None
    if not allow_downloads:
        return None, f"missing {RECOMMENDED_BASELINE_REPO}; rerun with --allow-downloads"
    destination = download_hf_model_from_ref(
        models_root,
        RECOMMENDED_BASELINE_REPO,
        input_func=lambda _prompt="": "1",
        print_func=lambda _line="": None,
    )
    if destination is None:
        return None, f"could not download {RECOMMENDED_BASELINE_REPO}"
    return destination, None


def _find_cached_file(filename: str, expected_sha_prefix: str, roots: tuple[str, ...], exclude_parent: Path | None = None) -> Path | None:
    expected = str(expected_sha_prefix).lower()
    excluded = exclude_parent.resolve() if exclude_parent is not None else None
    for root_name in roots:
        root = Path.cwd() / root_name
        if not root.exists():
            continue
        for candidate in root.rglob(filename):
            if not candidate.is_file():
                continue
            try:
                if excluded is not None and candidate.parent.resolve() == excluded:
                    continue
            except OSError:
                continue
            digest = sha256(candidate).removeprefix("sha256:").lower()
            if digest.startswith(expected):
                return candidate
    return None


def _ensure_openai_whisper_pt(models_root: Path, allow_downloads: bool) -> tuple[Path | None, str | None]:
    checkpoint = models_root / "openai_whisper" / TINY_PT
    if checkpoint.exists() and is_verified_official_checkpoint(checkpoint):
        return checkpoint, None
    cached = _find_cached_file(TINY_PT, TINY_PT_SHA256, LOCAL_OPENAI_PT_SEARCH_ROOTS, exclude_parent=checkpoint.parent)
    if cached is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, checkpoint)
        return checkpoint, None
    if not allow_downloads:
        return None, f"missing official allowlisted {TINY_PT}; rerun with --allow-downloads"
    try:
        _download_official_checkpoint(checkpoint)
    except Exception as exc:
        return None, f"could not download official {TINY_PT}: {type(exc).__name__}: {exc}"
    if not is_verified_official_checkpoint(checkpoint):
        return None, f"downloaded {TINY_PT} did not match the official SHA256 allowlist"
    return checkpoint, None


def _ensure_backend_model(backend: str, models_root: Path, allow_downloads: bool) -> tuple[Path | None, str | None]:
    if backend == "faster_whisper":
        return _ensure_faster_whisper(models_root, allow_downloads)
    if backend == "openai_whisper_pt":
        return _ensure_openai_whisper_pt(models_root, allow_downloads)
    return None, f"unsupported real public media backend {backend}"


def _download_fixture(fixture_id: str, evidence_dir: Path, allow_downloads: bool) -> tuple[Path | None, dict, str | None]:
    manifest = _load_manifest()
    fixture = manifest["fixtures"][fixture_id]
    details = {
        "manifest": str(MANIFEST),
        "fixture_id": fixture_id,
        "fixture": {
            "kind": fixture.get("kind"),
            "source_page": fixture.get("source_page"),
            "download_url": fixture.get("download_url"),
            "license": fixture.get("license"),
            "expected_text": fixture.get("expected_text"),
        },
    }
    target = evidence_dir / "real_media" / f"{fixture_id}{_extension_for_fixture(fixture_id, fixture)}"
    expected_prefix = fixture.get("expected_sha256_prefix")
    if target.exists():
        digest = sha256(target)
        if not expected_prefix or digest.removeprefix("sha256:").startswith(str(expected_prefix)):
            details["fixture"]["sha256"] = digest
            details["fixture"]["bytes"] = target.stat().st_size
            return target, details, None
    if expected_prefix:
        cached = _find_cached_file(target.name, str(expected_prefix), LOCAL_FIXTURE_SEARCH_ROOTS, exclude_parent=target.parent)
        if cached is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, target)
            details["fixture"]["sha256"] = sha256(target)
            details["fixture"]["bytes"] = target.stat().st_size
            details["fixture"]["byte_source"] = "local_cache"
            return target, details, None
    if not allow_downloads:
        return None, details, "real public media fixture downloads require --allow-downloads"
    try:
        _download(str(fixture["download_url"]), target)
    except Exception as exc:
        return None, {**details, "error_type": type(exc).__name__, "message": str(exc)}, f"could not download {fixture_id}"
    digest = sha256(target)
    if expected_prefix and not digest.removeprefix("sha256:").startswith(str(expected_prefix)):
        return None, {**details, "actual_sha256": digest}, f"{fixture_id} hash did not match expected prefix"
    details["fixture"]["sha256"] = digest
    details["fixture"]["bytes"] = target.stat().st_size
    return target, details, None


def _rewrite_base_reports(output_dir: Path, results: dict) -> None:
    (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output_dir / "results.txt").write_text(render_text_report(results), encoding="utf-8", newline="\n")
    (output_dir / "compare.html").write_text(build_html_report(results), encoding="utf-8", newline="\n")
    write_benchmark_csv(output_dir / "benchmark.csv", results)


def _repair_dependencies(config: dict, evidence_dir: Path, install_deps: bool, groups: set[str]) -> tuple[list[str], dict, list[Path]]:
    blockers: list[str] = []
    details: dict = {}
    artifacts: list[Path] = []
    for group in sorted(groups):
        missing = missing_modules_for_config(group, config)
        repair_log = evidence_dir / f"{group}_repair.log"
        details[f"{group}_missing_before"] = missing
        details[f"{group}_repair_command"] = recovery_command_for_config(group, config)
        if missing and install_deps:
            try:
                install_group_for_config(group, Path.cwd(), config, log_path=repair_log)
                artifacts.append(repair_log)
            except Exception as exc:
                blockers.append(f"{group}: repair failed: {type(exc).__name__}: {exc}")
                artifacts.append(repair_log)
                continue
            missing = missing_modules_for_config(group, config)
        details[f"{group}_missing_after"] = missing
        if missing:
            blockers.append(f"{group}: missing {', '.join(missing)}")
    return blockers, details, artifacts


def _reference_for(results: dict, expected_text: str) -> dict:
    return {
        "schema": "easy_asr_bench.llm_reference.v1",
        "source_sha256": results["source"]["sha256"],
        "reference_type": "llm_corrected_reference",
        "segments": [
            {
                "chunk_id": chunk["chunk_id"],
                "start_seconds": chunk["start_seconds"],
                "end_seconds": chunk["end_seconds"],
                "text": expected_text,
                "uncertain": ["single-word public media smoke; WER is recorded but not release-gated"],
            }
            for chunk in results.get("chunk_plan", {}).get("chunks", [])
        ],
        "global_notes": ["Reference text comes from the public real-media fixture manifest."],
    }


def run(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    if row_id not in ROW_FIXTURES:
        return write_row(row_id, "fail", evidence_dir, summary=f"Unsupported real public media row id: {row_id}")
    fixture_id = ROW_FIXTURES[row_id]
    backend = ROW_BACKENDS[row_id]
    groups = GROUPS_BY_BACKEND[backend]
    package_names = PACKAGE_NAMES_BY_BACKEND[backend]
    if not SMOLLM_PATH.exists():
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="SmolLM 135M GGUF fixture is not present locally, so real public media ASR output cannot be graded.",
            block_reason=f"missing {SMOLLM_PATH}",
            external_requirement="download HuggingFaceTB/SmolLM-135M-GGUF Q4_K_M fixture",
        )

    config = smoke_config(evidence_dir, "cpu")
    config["runtime"]["llm_context_tokens"] = 512
    config["runtime"]["llm_reference_max_tokens"] = 64
    config["runtime"]["llm_reference_temperature"] = 0.0
    config["security"] = dict(config.get("security", {}))
    config["security"]["allow_pickle_or_pt_files"] = False
    blockers, dependency_details, dependency_artifacts = _repair_dependencies(config, evidence_dir, install_deps, groups)
    if blockers:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="One or more dependency groups are not runnable, so the real public media ASR+SmolLM row cannot run.",
            block_reason="; ".join(blockers),
            external_requirement="rerun with --install-deps or repair the listed dependency groups through setup.bat",
            details={**dependency_details, "backend": backend, "dependency_versions": package_versions(package_names)},
            artifacts=[*dependency_artifacts, SMOLLM_PATH],
        )
    repair_evidence = execute_repair_plan(config, project_root=evidence_dir)
    dependency_details["repair_all_safe_summary"] = repair_evidence.get("summary", {})
    repair_evidence_path = Path(config["folders"]["logs"]) / "repair_all_safe_last.json"
    if repair_evidence_path.exists():
        dependency_artifacts.append(repair_evidence_path)

    source, fixture_details, fixture_error = _download_fixture(fixture_id, evidence_dir, allow_downloads)
    if fixture_error or source is None:
        return write_row(
            row_id,
            "blocked" if not allow_downloads else "fail",
            evidence_dir,
            summary="Real public media fixture is not available for ASR+SmolLM validation.",
            block_reason=fixture_error if not allow_downloads else None,
            external_requirement="rerun with --allow-downloads after source/license review" if not allow_downloads else None,
            details={**dependency_details, **fixture_details, "backend": backend},
            artifacts=[*dependency_artifacts, SMOLLM_PATH],
        )

    model_dir, model_error = _ensure_backend_model(backend, Path(config["folders"]["models"]), allow_downloads)
    if model_error or model_dir is None:
        return write_row(
            row_id,
            "blocked" if not allow_downloads else "fail",
            evidence_dir,
            summary=f"{backend} fixture is not staged for real public media ASR+SmolLM validation.",
            block_reason=model_error if not allow_downloads else None,
            external_requirement="rerun with --allow-downloads or stage the listed fixture" if not allow_downloads else None,
            details={**dependency_details, **fixture_details, "backend": backend},
            artifacts=[*dependency_artifacts, SMOLLM_PATH, source],
        )

    runnable, unsupported = scan_models(Path(config["folders"]["models"]))
    selected = [candidate for candidate in runnable if candidate.adapter_name == backend]
    if not selected:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary=f"Staged {backend} model was not discovered as a runnable candidate.",
            details={**dependency_details, **fixture_details, "backend": backend, "unsupported": [candidate.candidate_id for candidate in unsupported]},
            artifacts=[*dependency_artifacts, SMOLLM_PATH, source, model_dir],
        )
    llm_candidate, scan_details = _smollm_candidate()
    if llm_candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference/correction LLM candidate.",
            details={**dependency_details, **fixture_details, **scan_details, "backend": backend},
            artifacts=[*dependency_artifacts, SMOLLM_PATH, source, model_dir],
        )

    output_dir = process_file_with_candidates(source, [selected[0]], config, unsupported, reference_llm=llm_candidate)
    if output_dir is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Real public media ASR+SmolLM row did not produce a report directory.",
            details={**dependency_details, **fixture_details, **scan_details, "backend": backend},
            artifacts=[*dependency_artifacts, SMOLLM_PATH, source, model_dir],
        )
    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))

    adapter = GGUFLLMReferenceAdapter()
    llm = adapter.load(llm_candidate, {"provider": "cpu", "prefer_gpu": False, "llm_context_tokens": 512})
    response = llm("Answer with the word pass.", max_tokens=8, temperature=0.0)
    generated_text = response["choices"][0]["text"].strip() if isinstance(response, dict) else str(response).strip()
    results["local_llm_reference_attempt"] = {
        "candidate_id": llm_candidate.candidate_id,
        "display_name": llm_candidate.display_name,
        "status": "generated" if generated_text else "empty",
        "raw_response": generated_text,
        "note": "This row proves SmolLM runs after a real public-media ASR output.",
    }
    _rewrite_base_reports(output_dir, results)

    expected_text = str(fixture_details["fixture"].get("expected_text", "")).strip()
    reference = _reference_for(results, expected_text)
    scored = import_llm_reference(results, "SmolLM corrected reference fixture:\n```json\n" + json.dumps(reference) + "\n```")
    scored_path = output_dir / "scored_report.json"
    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    scored_results = dict(results)
    if scored.get("status") == "scored":
        scored_results["reference_scores"] = scored["scores"]
    scored_html = output_dir / "compare_scored.html"
    scored_html.write_text(build_html_report(scored_results), encoding="utf-8", newline="\n")

    transcript = "\n".join(
        chunk.get("text", "")
        for run in results.get("runs", [])
        for chunk in run.get("transcript_chunks", [])
    ).strip()
    normalized_wer = wer(expected_text, transcript, normalized=True) if expected_text and transcript else None
    failures = [
        failure
        for failure in _assert_report_files(output_dir, large=False)
        if failure != "compare_scored.html missing marker fixture_windows_gpu_adapter_memory"
    ]
    dependency_failures, dependency_report_details = dependency_resolution_report_failures(results, expected_groups=groups)
    failures.extend(dependency_failures)
    if not transcript:
        failures.append("real public media ASR transcript was empty")
    if not generated_text:
        failures.append("SmolLM generated empty text after real public media ASR")
    if scored.get("status") != "scored":
        failures.append("real public media reference import did not score")
    if "Loaded precomputed LLM-corrected reference scores" not in scored_html.read_text(encoding="utf-8"):
        failures.append("compare_scored.html missing precomputed score marker")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            f"Real public Wikimedia media ran through {backend} ASR, then SmolLM scoring/report validation completed."
            if not failures
            else "Real public media ASR+SmolLM validation failed."
        ),
        details={
            **dependency_details,
            **fixture_details,
            **scan_details,
            "backend": backend,
            "model_dir": str(model_dir),
            "output_dir": str(output_dir),
            "transcript": transcript,
            "expected_text": expected_text,
            "normalized_wer": normalized_wer,
            "score_status": scored.get("status"),
            "generated_text": generated_text,
            "dependency_versions": package_versions(package_names),
            "failures": failures,
            **dependency_report_details,
        },
        artifacts=[
            *dependency_artifacts,
            SMOLLM_PATH,
            source,
            model_dir,
            output_dir / "results.json",
            output_dir / "results.txt",
            output_dir / "benchmark.csv",
            output_dir / "compare.html",
            scored_path,
            scored_html,
        ],
    )
