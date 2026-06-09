from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

from app.adapters.openai_whisper_pt import is_verified_official_checkpoint
from app.config import load_config
from app.dependency_manager import cuda_diagnostics, install_group_for_config, missing_modules_for_config, recovery_command_for_config
from app.hf_model_downloader import RECOMMENDED_BASELINE_REPO, download_hf_model_from_ref
from app.html_report_builder import build_html_report
from app.main import process_file_with_candidates
from app.model_scanner import scan_models
from app.reference_import import import_llm_reference
from app.repair_plan import execute_repair_plan
from app.scoring import wer
from qa.run_real_tiny_model_smoke import REFERENCE_TEXT, generate_windows_sapi_wav, smoke_config
from qa.runtime_matrix.common import dependency_resolution_report_failures, package_versions, write_row
from qa.runtime_matrix.rows.generic_onnx_smollm_grading import GENERIC_ONNX_QUALITY_MODEL_FILE
from qa.runtime_matrix.rows.generic_onnx_smollm_grading import GENERIC_ONNX_QUALITY_MODEL_URL
from qa.runtime_matrix.rows.generic_onnx_smollm_grading import GENERIC_ONNX_QUALITY_REPO
from qa.runtime_matrix.rows.generic_onnx_smollm_grading import GENERIC_ONNX_QUALITY_VOCAB_URL
from qa.runtime_matrix.rows.generic_onnx_smollm_grading import _copy_cached_public_ctc_fixture
from qa.runtime_matrix.rows.generic_onnx_smollm_grading import _download as _download_generic_onnx_file
from qa.runtime_matrix.rows.generic_onnx_smollm_grading import _find_cached_public_ctc_fixture
from qa.runtime_matrix.rows.generic_onnx_smollm_grading import _write_public_ctc_manifest
from qa.runtime_matrix.rows.gguf_asr_mmproj import PUBLIC_QWEN3_ASR_MMPROJ_FILE
from qa.runtime_matrix.rows.gguf_asr_mmproj import PUBLIC_QWEN3_ASR_MMPROJ_URL
from qa.runtime_matrix.rows.gguf_asr_mmproj import PUBLIC_QWEN3_ASR_MODEL_FILE
from qa.runtime_matrix.rows.gguf_asr_mmproj import PUBLIC_QWEN3_ASR_MODEL_URL
from qa.runtime_matrix.rows.gguf_asr_mmproj import PUBLIC_QWEN3_ASR_REPO
from qa.runtime_matrix.rows.gguf_asr_mmproj import _copy_cached_public_fixture as _copy_cached_gguf_asr_fixture
from qa.runtime_matrix.rows.gguf_asr_mmproj import _download as _download_gguf_asr_file
from qa.runtime_matrix.rows.gguf_asr_mmproj import _find_cached_public_fixture as _find_cached_gguf_asr_fixture
from qa.runtime_matrix.rows.gguf_asr_mmproj import _write_manifest as _write_gguf_asr_manifest
from qa.runtime_matrix.rows.hf_safetensors_tiny import HF_CTC_REPO, HF_WHISPER_REPO, _download_repo
from qa.runtime_matrix.rows.openai_whisper_pt_safety import TINY_PT, TINY_PT_SHA256, TINY_PT_URL, _download_official_checkpoint
from qa.runtime_matrix.rows.real_public_media_faster_whisper_smollm import _find_cached_file as _find_cached_public_media_file
from qa.runtime_matrix.rows.smollm_reference_grading_report import SMOLLM_PATH, _smollm_candidate
from qa.runtime_matrix.rows.whisper_cpp_ggml import MODEL_FILE as WHISPER_CPP_MODEL_FILE
from qa.runtime_matrix.rows.whisper_cpp_ggml import MODEL_SHA256 as WHISPER_CPP_MODEL_SHA256
from qa.runtime_matrix.rows.whisper_cpp_ggml import MODEL_URL as WHISPER_CPP_MODEL_URL
from qa.runtime_matrix.rows.whisper_cpp_ggml import _download_model as _download_whisper_cpp_model
from qa.runtime_matrix.rows.whisper_cpp_smollm_grading import _find_cached_model as _find_cached_whisper_cpp_model


GROUPS = ["python_packaging", "media_tools", "faster_whisper", "onnx", "transformers_cpu", "whisper_cpp", "openai_whisper", "llama_cpp", "llama_mtmd"]
REQUIRED_ADAPTERS = {
    "faster_whisper",
    "openai_whisper_pt",
    "generic_onnx_manifest",
    "hf_whisper_asr",
    "hf_transformers_asr",
    "whisper_cpp",
    "gguf_asr_mmproj",
}
QUALITY_ADAPTERS = {"faster_whisper", "openai_whisper_pt", "generic_onnx_manifest", "gguf_asr_mmproj"}
LOCAL_REPO_FIXTURE_SEARCH_ROOTS = ("Temp", "Models", "Cache")
LOCAL_OPENAI_PT_SEARCH_ROOTS = ("Temp", "Models", "Cache")


def _repo_folder_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _copy_repo_folder(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    ignore = shutil.ignore_patterns(".hf_cache", "__pycache__")
    shutil.copytree(source, destination, ignore=ignore)


def _find_cached_repo_folder(repo_id: str, required_names: set[str], exclude: Path | None = None) -> Path | None:
    folder_name = _repo_folder_name(repo_id)
    excluded = exclude.resolve() if exclude is not None and exclude.exists() else None
    for root_name in LOCAL_REPO_FIXTURE_SEARCH_ROOTS:
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
    for root_name in LOCAL_REPO_FIXTURE_SEARCH_ROOTS:
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


def _copy_cached_repo_if_available(repo_id: str, destination: Path, required_names: set[str]) -> bool:
    cached = _find_cached_repo_folder(repo_id, required_names, exclude=destination)
    if cached is None:
        return False
    _copy_repo_folder(cached, destination)
    return True


def _repair_dependencies(config: dict, evidence_dir: Path, install_deps: bool) -> tuple[list[str], dict, list[Path]]:
    blockers: list[str] = []
    details: dict = {}
    artifacts: list[Path] = []
    for group in GROUPS:
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


def _ensure_faster_whisper(models_root: Path, allow_downloads: bool) -> tuple[Path | None, str | None]:
    runnable, _ = scan_models(models_root)
    candidates = [candidate for candidate in runnable if candidate.adapter_name == "faster_whisper"]
    if candidates:
        return candidates[0].path, None
    destination = models_root / _repo_folder_name(RECOMMENDED_BASELINE_REPO)
    cached = _find_cached_repo_folder(RECOMMENDED_BASELINE_REPO, {"model.bin", "config.json"}, exclude=destination)
    if cached is None:
        cached = _find_cached_model_folder(
            {"model.bin", "config.json"},
            ({"tokenizer.json", "vocabulary.json", "vocabulary.txt", "vocab.json"},),
            exclude=destination,
        )
    if cached is not None:
        _copy_repo_folder(cached, destination)
        runnable, _ = scan_models(models_root)
        candidates = [candidate for candidate in runnable if candidate.adapter_name == "faster_whisper"]
        if candidates:
            return candidates[0].path, None
    if not allow_downloads:
        return None, f"missing {RECOMMENDED_BASELINE_REPO}; rerun with --allow-downloads"
    destination = download_hf_model_from_ref(models_root, RECOMMENDED_BASELINE_REPO, input_func=lambda _prompt="": "1", print_func=lambda _line="": None)
    if destination is None:
        return None, f"could not download {RECOMMENDED_BASELINE_REPO}"
    return destination, None


def _ensure_hf_safetensors(models_root: Path, repo_id: str, allow_downloads: bool) -> tuple[Path | None, str | None]:
    model_dir = models_root / repo_id.replace("/", "__")
    if list(model_dir.glob("*.safetensors")):
        return model_dir, None
    if _copy_cached_repo_if_available(repo_id, model_dir, {"config.json"}):
        if list(model_dir.glob("*.safetensors")):
            return model_dir, None
    if not allow_downloads:
        return None, f"missing {repo_id}; rerun with --allow-downloads"
    try:
        _download_repo(repo_id, model_dir)
    except Exception as exc:
        return None, f"could not download {repo_id}: {type(exc).__name__}: {exc}"
    return model_dir, None


def _ensure_whisper_cpp(models_root: Path, allow_downloads: bool) -> tuple[Path | None, str | None]:
    model_path = models_root / "whisper_cpp" / WHISPER_CPP_MODEL_FILE
    if model_path.exists():
        return model_path, None
    cached = _find_cached_whisper_cpp_model(WHISPER_CPP_MODEL_FILE, WHISPER_CPP_MODEL_SHA256)
    if cached is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, model_path)
        return model_path, None
    if not allow_downloads:
        return None, f"missing {WHISPER_CPP_MODEL_FILE}; rerun with --allow-downloads"
    try:
        _download_whisper_cpp_model(model_path)
    except Exception as exc:
        return None, f"could not download {WHISPER_CPP_MODEL_URL}: {type(exc).__name__}: {exc}"
    return model_path, None


def _ensure_openai_pt(models_root: Path, allow_downloads: bool) -> tuple[Path | None, str | None]:
    model_path = models_root / "openai_whisper" / TINY_PT
    if model_path.exists() and is_verified_official_checkpoint(model_path):
        return model_path, None
    cached = _find_cached_public_media_file(TINY_PT, TINY_PT_SHA256, LOCAL_OPENAI_PT_SEARCH_ROOTS, exclude_parent=model_path.parent)
    if cached is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, model_path)
        return model_path, None
    if not allow_downloads:
        return None, f"missing official allowlisted {TINY_PT}; rerun with --allow-downloads"
    try:
        _download_official_checkpoint(model_path)
    except Exception as exc:
        return None, f"could not download {TINY_PT_URL}: {type(exc).__name__}: {exc}"
    return model_path, None


def _ensure_generic_onnx_quality(models_root: Path, allow_downloads: bool) -> tuple[Path | None, str | None, list[Path]]:
    model_dir = models_root / GENERIC_ONNX_QUALITY_REPO.replace("/", "__")
    model_path = model_dir / "model.onnx"
    vocab_path = model_dir / "vocab.json"
    manifest_path = model_dir / "modelbench.json"
    if model_path.exists() and vocab_path.exists():
        _write_public_ctc_manifest(model_dir)
        return model_dir, None, [model_path, vocab_path, manifest_path]
    cached = _find_cached_public_ctc_fixture()
    if cached is not None:
        _copy_cached_public_ctc_fixture(cached, model_dir)
        return model_dir, None, [model_path, vocab_path, manifest_path]
    if not allow_downloads:
        return None, f"missing {GENERIC_ONNX_QUALITY_REPO} {GENERIC_ONNX_QUALITY_MODEL_FILE}; rerun with --allow-downloads", [model_path, vocab_path, manifest_path]
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        _download_generic_onnx_file(GENERIC_ONNX_QUALITY_MODEL_URL, model_path)
        _download_generic_onnx_file(GENERIC_ONNX_QUALITY_VOCAB_URL, vocab_path)
        _write_public_ctc_manifest(model_dir)
    except Exception as exc:
        return None, f"could not download {GENERIC_ONNX_QUALITY_REPO} {GENERIC_ONNX_QUALITY_MODEL_FILE}: {type(exc).__name__}: {exc}", [model_path, vocab_path, manifest_path]
    return model_dir, None, [model_path, vocab_path, manifest_path]


def _ensure_gguf_asr_mmproj(models_root: Path, allow_downloads: bool) -> tuple[Path | None, str | None, list[Path]]:
    model_dir = models_root / PUBLIC_QWEN3_ASR_REPO.replace("/", "__")
    model_path = model_dir / PUBLIC_QWEN3_ASR_MODEL_FILE
    mmproj_path = model_dir / PUBLIC_QWEN3_ASR_MMPROJ_FILE
    manifest_path = _write_gguf_asr_manifest(model_dir)
    artifacts = [model_path, mmproj_path, manifest_path]
    if model_path.exists() and mmproj_path.exists():
        return model_dir, None, artifacts
    cached = _find_cached_gguf_asr_fixture(model_dir)
    if cached is not None:
        _copy_cached_gguf_asr_fixture(cached, model_dir)
        return model_dir, None, artifacts
    if not allow_downloads:
        return None, f"missing {PUBLIC_QWEN3_ASR_REPO} {PUBLIC_QWEN3_ASR_MODEL_FILE} plus {PUBLIC_QWEN3_ASR_MMPROJ_FILE}; rerun with --allow-downloads", artifacts
    try:
        _download_gguf_asr_file(PUBLIC_QWEN3_ASR_MODEL_URL, model_path)
        _download_gguf_asr_file(PUBLIC_QWEN3_ASR_MMPROJ_URL, mmproj_path)
        _write_gguf_asr_manifest(model_dir)
    except Exception as exc:
        return None, f"could not download {PUBLIC_QWEN3_ASR_REPO} GGUF ASR fixture: {type(exc).__name__}: {exc}", artifacts
    return model_dir, None, artifacts


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
        "global_notes": ["Reference text is the generated SAPI prompt used by this same-media multi-model runtime row."],
    }


def _write_scored_artifacts(results: dict, output_dir: Path) -> tuple[dict, Path]:
    reference = _reference_for(results, REFERENCE_TEXT)
    scored = import_llm_reference(results, "SmolLM corrected reference fixture:\n```json\n" + json.dumps(reference) + "\n```")
    scored_path = output_dir / "scored_report.json"
    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    scored_html = output_dir / "compare_scored.html"
    scored_html.write_text(build_html_report(scored), encoding="utf-8", newline="\n")
    return scored, scored_html


def _provider_for_row(row_id: str) -> str:
    if row_id.endswith("_directml"):
        return "directml"
    return "cpu"


def run(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    if row_id not in {"same_media_multi_model_smollm_benchmark", "same_media_multi_model_smollm_benchmark_directml"}:
        return write_row(row_id, "fail", evidence_dir, summary=f"Unhandled same-media multi-model row: {row_id}")
    if not SMOLLM_PATH.exists():
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="SmolLM 135M GGUF fixture is not present locally, so the same-media benchmark cannot run the local grading path.",
            block_reason=f"missing {SMOLLM_PATH}",
            external_requirement="download HuggingFaceTB/SmolLM-135M-GGUF Q4_K_M fixture",
        )

    provider = _provider_for_row(row_id)
    diagnostics = cuda_diagnostics()
    config = smoke_config(evidence_dir, provider)
    config["security"] = dict(config.get("security", {}))
    config["security"]["allow_pickle_or_pt_files"] = False
    config["runtime"]["llm_context_tokens"] = 1024
    config["runtime"]["llm_reference_max_tokens"] = 128
    config["runtime"]["llm_reference_temperature"] = 0.0
    blockers, dependency_details, dependency_artifacts = _repair_dependencies(config, evidence_dir, install_deps)
    dependency_details["requested_provider"] = provider
    dependency_details["cuda_provider_checks"] = diagnostics
    try:
        from app.dependency_manager import prepare_llama_cpp_dll_search_path

        prepare_llama_cpp_dll_search_path()
        import llama_cpp  # noqa: F401
    except ModuleNotFoundError:
        blockers.append("llama_cpp: missing llama_cpp import for CPU SmolLM grading")
        dependency_details["llama_cpp_missing_after"] = ["llama_cpp"]
        dependency_details["llama_cpp_repair_command"] = recovery_command_for_config("llama_cpp", config)
    if blockers:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="One or more dependency groups are not runnable, so the same-media multi-model benchmark cannot run.",
            block_reason="; ".join(blockers),
            external_requirement="rerun with --install-deps or repair the listed dependency groups through setup.bat",
            details={**dependency_details, "dependency_versions": package_versions(["faster-whisper", "ctranslate2", "onnx", "onnxruntime", "onnxruntime-directml", "torch", "transformers", "safetensors", "pywhispercpp", "openai-whisper", "llama-cpp-python"])},
            artifacts=dependency_artifacts,
        )
    if provider == "directml" and "DmlExecutionProvider" not in cuda_diagnostics().get("onnxruntime_providers", []):
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Same-media DirectML benchmark requires ONNX Runtime DmlExecutionProvider, which is not visible.",
            block_reason="DmlExecutionProvider missing from onnxruntime providers after dependency repair checks",
            external_requirement="Windows DirectML-capable GPU with onnxruntime-directml installed and provider visible",
            details={**dependency_details, "dependency_versions": package_versions(["onnxruntime", "onnxruntime-directml"])},
            artifacts=dependency_artifacts,
        )
    repair_evidence = execute_repair_plan(config, project_root=evidence_dir)
    dependency_details["repair_all_safe_summary"] = repair_evidence.get("summary", {})
    repair_evidence_path = Path(config["folders"]["logs"]) / "repair_all_safe_last.json"
    if repair_evidence_path.exists():
        dependency_artifacts.append(repair_evidence_path)

    models_root = Path(config["folders"]["models"])
    fixture_errors: list[str] = []
    fixture_artifacts: list[Path] = []
    for path, error in [
        _ensure_faster_whisper(models_root, allow_downloads),
        _ensure_hf_safetensors(models_root, HF_WHISPER_REPO, allow_downloads),
        _ensure_hf_safetensors(models_root, HF_CTC_REPO, allow_downloads),
        _ensure_whisper_cpp(models_root, allow_downloads),
        _ensure_openai_pt(models_root, allow_downloads),
    ]:
        if path is not None:
            fixture_artifacts.append(path)
        if error:
            fixture_errors.append(error)
    for fixture_path, fixture_error, artifacts in [
        _ensure_generic_onnx_quality(models_root, allow_downloads),
        _ensure_gguf_asr_mmproj(models_root, allow_downloads),
    ]:
        if fixture_path is not None:
            fixture_artifacts.append(fixture_path)
        if fixture_error:
            fixture_errors.append(fixture_error)
        fixture_artifacts.extend(artifacts)
    if fixture_errors:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="One or more model fixtures are not staged for the same-media multi-model benchmark.",
            block_reason="; ".join(fixture_errors),
            external_requirement="rerun with --allow-downloads or stage the listed fixtures",
            details=dependency_details,
            artifacts=[*dependency_artifacts, *fixture_artifacts, SMOLLM_PATH],
        )

    runnable, unsupported = scan_models(models_root)
    selected = []
    adapter_counts: dict[str, int] = {}
    for adapter_name in ["faster_whisper", "openai_whisper_pt", "generic_onnx_manifest", "hf_whisper_asr", "hf_transformers_asr", "whisper_cpp", "gguf_asr_mmproj"]:
        matches = [candidate for candidate in runnable if candidate.adapter_name == adapter_name]
        unsupported_matches = [candidate for candidate in unsupported if candidate.adapter_name == adapter_name]
        if adapter_name == "gguf_asr_mmproj" and unsupported_matches:
            preferred = [candidate for candidate in unsupported_matches if PUBLIC_QWEN3_ASR_REPO.replace("/", "__").lower() in str(candidate.path).lower()]
            candidate = preferred[0] if preferred else unsupported_matches[0]
            selected.append(
                replace(
                    candidate,
                    runnable=True,
                    runnable_after_dependency_install=True,
                    warnings=[*candidate.warnings, "Runtime matrix forced this recognized-experimental candidate into the same-media live benchmark."],
                )
            )
            adapter_counts[adapter_name] = len(unsupported_matches)
        elif matches:
            if adapter_name == "generic_onnx_manifest":
                preferred = [candidate for candidate in matches if GENERIC_ONNX_QUALITY_REPO.replace("/", "__").lower() in str(candidate.path).lower()]
                selected.append(preferred[0] if preferred else matches[0])
            else:
                selected.append(matches[0])
            adapter_counts[adapter_name] = len(matches)
        else:
            adapter_counts[adapter_name] = 0
    missing_adapters = sorted(REQUIRED_ADAPTERS - {candidate.adapter_name for candidate in selected})
    if missing_adapters:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="The same-media benchmark did not discover every required runnable adapter.",
            details={
                **dependency_details,
                "adapter_counts": adapter_counts,
                "missing_adapters": missing_adapters,
                "runnable": [candidate.adapter_name for candidate in runnable],
                "unsupported": [{"adapter_name": candidate.adapter_name, "warnings": candidate.warnings, "missing": candidate.missing_files} for candidate in unsupported],
            },
            artifacts=[*dependency_artifacts, *fixture_artifacts, SMOLLM_PATH],
        )
    llm_candidate, scan_details = _smollm_candidate()
    if llm_candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference/correction LLM candidate.",
            details={**dependency_details, **scan_details},
            artifacts=[*dependency_artifacts, *fixture_artifacts, SMOLLM_PATH],
        )

    source = Path(config["folders"]["input"]) / "same_media_multi_model_sapi.wav"
    generate_windows_sapi_wav(source, REFERENCE_TEXT)
    config["transcription"] = dict(config.get("transcription", {}))
    config["transcription"]["ar_prompt"] = "Transcribe the audio. Return only the transcript."
    config["transcription"]["ar_max_new_tokens"] = 128
    config["transcription"]["temperature"] = 0.0
    config["llama_cpp"] = dict(config.get("llama_cpp", {}))
    config["llama_cpp"]["timeout_seconds"] = 900
    output_dir = process_file_with_candidates(source, selected, config, unsupported, reference_llm=llm_candidate)
    if output_dir is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Same-media multi-model benchmark did not produce a report directory.",
            details={**dependency_details, **scan_details, "selected_adapters": [candidate.adapter_name for candidate in selected]},
            artifacts=[*dependency_artifacts, *fixture_artifacts, SMOLLM_PATH, source],
        )

    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    scored, scored_html = _write_scored_artifacts(results, output_dir)
    dependency_report_failures, dependency_report_details = dependency_resolution_report_failures(results, expected_groups=set(GROUPS))
    runs = results.get("runs", [])
    run_adapters = {run["model"]["adapter_name"] for run in runs}
    failures: list[str] = list(dependency_report_failures)
    if run_adapters != REQUIRED_ADAPTERS:
        failures.append("report run adapters did not match required adapters: " + ", ".join(sorted(run_adapters)))
    for run in runs:
        adapter_name = run["model"]["adapter_name"]
        text = "\n".join(chunk.get("text", "") for chunk in run.get("transcript_chunks", []))
        if adapter_name in QUALITY_ADAPTERS:
            normalized_wer = wer(REFERENCE_TEXT, text, normalized=True) if text.strip() else 1.0
            if normalized_wer > 0.85:
                failures.append(f"{adapter_name} normalized WER {normalized_wer:.3f} exceeded threshold 0.850")
        if provider == "directml" and adapter_name == "generic_onnx_manifest":
            provider_summary = run.get("metrics", {}).get("provider_summary", {})
            active_providers = provider_summary.get("active_providers", [])
            if "DmlExecutionProvider" not in active_providers:
                failures.append("generic_onnx_manifest did not run with active DmlExecutionProvider")
    if scored.get("status") != "scored":
        failures.append("same-media scored reference status was not scored")
    scores = scored.get("scores", {})
    if set(scores) != {run["model"]["candidate_id"] for run in runs}:
        failures.append("scored report did not include every run candidate")
    for name in ["results.json", "results.txt", "benchmark.csv", "compare.html", "scored_report.json", "compare_scored.html"]:
        if not (output_dir / name).exists():
            failures.append(f"missing report artifact {name}")
    scored_html_text = scored_html.read_text(encoding="utf-8")
    if "Loaded precomputed LLM-corrected reference scores" not in scored_html_text:
        failures.append("compare_scored.html missing precomputed score marker")

    transcripts = {
        run["model"]["adapter_name"]: "\n".join(chunk.get("text", "") for chunk in run.get("transcript_chunks", []))
        for run in runs
    }
    quality_wers = {
        adapter_name: wer(REFERENCE_TEXT, text, normalized=True) if text.strip() else 1.0
        for adapter_name, text in transcripts.items()
        if adapter_name in QUALITY_ADAPTERS
    }
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            (
                "Same-media multi-model DirectML benchmark ran the mixed ASR fixture set, verified the Generic ONNX leg used DirectML, then SmolLM scoring/report validation completed."
                if provider == "directml"
                else "Same-media multi-model benchmark ran all locally runnable CPU-format fixtures, then SmolLM scoring/report validation completed."
            )
            if not failures
            else "Same-media multi-model benchmark validation failed."
        ),
        details={
            **dependency_details,
            **scan_details,
            "selected_adapters": [candidate.adapter_name for candidate in selected],
            "provider": provider,
            "adapter_counts": adapter_counts,
            "run_adapters": sorted(run_adapters),
            "reference_text": REFERENCE_TEXT,
            "quality_wers": quality_wers,
            "transcripts": transcripts,
            "score_status": scored.get("status"),
            "score_count": len(scores),
            "output_dir": str(output_dir),
            **dependency_report_details,
            "dependency_versions": package_versions(["pip", "setuptools", "faster-whisper", "ctranslate2", "onnx", "onnxruntime", "onnxruntime-directml", "torch", "transformers", "safetensors", "pywhispercpp", "openai-whisper", "llama-cpp-python"]),
            "failures": failures,
        },
        artifacts=[
            *dependency_artifacts,
            *fixture_artifacts,
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
