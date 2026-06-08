from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.dependency_manager import install_group_for_config, missing_modules_for_config, recovery_command_for_config
from app.html_report_builder import build_html_report
from app.main import process_file_with_candidates
from app.model_scanner import scan_models
from app.reference_import import import_llm_reference
from app.results_writer import build_results, write_all_reports
from app.scoring import wer
from qa.run_real_tiny_model_smoke import REFERENCE_TEXT, generate_windows_sapi_wav, smoke_config
from qa.runtime_matrix.common import dependency_resolution_report_failures, package_versions, write_row
from qa.runtime_matrix.rows.hf_safetensors_tiny import _download_repo, _repo_for_row
from qa.runtime_matrix.rows.smollm_reference_grading_report import SMOLLM_PATH, _smollm_candidate

HF_WHISPER_QUALITY_REPO = "openai/whisper-tiny"
HF_CTC_QUALITY_REPO = "facebook/wav2vec2-base-960h"
HF_WHISPER_SHARDED_REPO = "optimum-internal-testing/tiny-random-whisper"


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
                "uncertain": ["structural tiny Safetensors fixture; not a quality-bearing human reference"],
            }
            for chunk in results.get("chunk_plan", {}).get("chunks", [])
        ],
        "global_notes": ["Reference text is derived from the tiny fixture output for stable structural scoring only."],
    }


def _ensure_transformers_deps(row_id: str, evidence_dir: Path, install_deps: bool, artifacts: list[Path]) -> dict | None:
    config = smoke_config(evidence_dir, "cpu")
    missing = missing_modules_for_config("transformers_cpu", config)
    if not missing:
        return None
    repair_log = evidence_dir / "transformers_cpu_repair.log"
    repair_command = recovery_command_for_config("transformers_cpu", config)
    if install_deps:
        try:
            install_group_for_config("transformers_cpu", Path.cwd(), config, log_path=repair_log)
        except Exception as exc:
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary="Transformers/Safetensors dependency repair failed.",
                block_reason=f"{type(exc).__name__}: {exc}",
                external_requirement=repair_command,
                details={"missing_before": missing, "dependency_versions": package_versions(["torch", "transformers", "safetensors", "sentencepiece", "tokenizers", "torchaudio"])},
                artifacts=[*artifacts, repair_log],
            )
        missing = missing_modules_for_config("transformers_cpu", config)
    if missing:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Transformers/Safetensors dependency group is not currently runnable.",
            block_reason=", ".join(missing),
            external_requirement=repair_command,
            details={"missing": missing, "dependency_versions": package_versions(["torch", "transformers", "safetensors", "sentencepiece", "tokenizers", "torchaudio"])},
            artifacts=[*artifacts, repair_log],
        )
    return None


def _ensure_repo(row_id: str, evidence_dir: Path, repo_id: str, allow_downloads: bool, artifacts: list[Path]) -> dict | Path:
    model_dir = evidence_dir / "Models" / repo_id.replace("/", "__")
    if list(model_dir.glob("*.safetensors")):
        return model_dir
    if not allow_downloads:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary=f"{repo_id} Safetensors fixture is not cached locally.",
            block_reason=f"missing local fixture {model_dir}",
            external_requirement=f"rerun with --allow-downloads to download {repo_id}",
            details={"repo_id": repo_id},
            artifacts=artifacts,
        )
    try:
        _download_repo(repo_id, model_dir)
    except Exception as exc:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary=f"Could not download {repo_id} Safetensors fixture.",
            block_reason=f"{type(exc).__name__}: {exc}",
            external_requirement=f"network access to https://huggingface.co/{repo_id}",
            details={"repo_id": repo_id},
            artifacts=artifacts,
        )
    return model_dir


def _materialize_sharded_whisper(source_dir: Path, sharded_dir: Path) -> list[Path]:
    index_files = list(sharded_dir.glob("*.safetensors.index*.json"))
    shard_files = list(sharded_dir.glob("*.safetensors"))
    if index_files and len(shard_files) > 1:
        return [*index_files, *shard_files]

    from transformers import AutoModelForSpeechSeq2Seq

    if sharded_dir.exists():
        shutil.rmtree(sharded_dir)
    sharded_dir.mkdir(parents=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(source_dir, local_files_only=True)
    model.save_pretrained(sharded_dir, safe_serialization=True, max_shard_size="50KB")
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if lower_name.endswith(".safetensors") or ".safetensors.index" in lower_name:
            continue
        if ".hf_cache" in path.parts:
            continue
        target = sharded_dir / path.relative_to(source_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
    return [*sharded_dir.glob("*.safetensors.index*.json"), *sharded_dir.glob("*.safetensors")]


def _run_whisper_quality(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    dependency_block = _ensure_transformers_deps(row_id, evidence_dir, install_deps, [SMOLLM_PATH])
    if dependency_block is not None:
        return dependency_block
    model_dir_or_row = _ensure_repo(row_id, evidence_dir, HF_WHISPER_QUALITY_REPO, allow_downloads, [SMOLLM_PATH])
    if isinstance(model_dir_or_row, dict):
        return model_dir_or_row
    model_dir = model_dir_or_row
    config = smoke_config(evidence_dir, "cpu")
    config["runtime"]["llm_context_tokens"] = 1024
    config["runtime"]["llm_reference_max_tokens"] = 128
    config["runtime"]["llm_reference_temperature"] = 0.0
    runnable, unsupported = scan_models(model_dir)
    candidates = [candidate for candidate in runnable if candidate.adapter_name == "hf_whisper_asr"]
    if not candidates:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary=f"{HF_WHISPER_QUALITY_REPO} did not scan as a runnable HF Whisper Safetensors model.",
            details={
                "repo_id": HF_WHISPER_QUALITY_REPO,
                "runnable": [candidate.adapter_name for candidate in runnable],
                "unsupported": [{"adapter_name": candidate.adapter_name, "missing": candidate.missing_files, "warnings": candidate.warnings} for candidate in unsupported],
            },
            artifacts=[SMOLLM_PATH, *model_dir.glob("*.safetensors")],
        )
    llm_candidate, scan_details = _smollm_candidate()
    if llm_candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference/correction LLM candidate.",
            details=scan_details,
            artifacts=[SMOLLM_PATH, *model_dir.glob("*.safetensors")],
        )
    source = Path(config["folders"]["input"]) / "hf_whisper_safetensors_quality_sapi.wav"
    generate_windows_sapi_wav(source, REFERENCE_TEXT)
    output_dir = process_file_with_candidates(source, [candidates[0]], config, unsupported, reference_llm=llm_candidate)
    if output_dir is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="HF Whisper Safetensors quality row did not produce a report directory.",
            details={"repo_id": HF_WHISPER_QUALITY_REPO, **scan_details},
            artifacts=[SMOLLM_PATH, *model_dir.glob("*.safetensors"), source],
        )
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
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

    dependency_report_failures, dependency_report_details = dependency_resolution_report_failures(results, expected_groups={"transformers_cpu", "llama_cpp"})
    failures: list[str] = list(dependency_report_failures)
    if not transcript.strip():
        failures.append("HF Whisper Safetensors speech transcript was empty")
    if normalized_wer > 0.85:
        failures.append(f"HF Whisper Safetensors normalized WER {normalized_wer:.3f} exceeded threshold 0.850")
    for name in ["results.json", "results.txt", "benchmark.csv", "compare.html", "scored_report.json", "compare_scored.html"]:
        if not (output_dir / name).exists():
            failures.append(f"missing report artifact {name}")
    scored_html_text = scored_html.read_text(encoding="utf-8")
    if "Loaded precomputed LLM-corrected reference scores" not in scored_html_text:
        failures.append("compare_scored.html missing precomputed score marker")
    run_id = runs[0]["model"]["candidate_id"] if runs else ""
    score = scored.get("scores", {}).get(run_id, {})
    if scored.get("status") != "scored" or score.get("normalized_wer") is None:
        failures.append("HF Whisper Safetensors scored reference was not produced")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "HF Whisper Safetensors transcribed generated speech with acceptable WER, then SmolLM scoring/report validation completed."
            if not failures
            else "HF Whisper Safetensors quality SmolLM grading validation failed."
        ),
        details={
            "repo_id": HF_WHISPER_QUALITY_REPO,
            "adapter_name": "hf_whisper_asr",
            "reference_text": REFERENCE_TEXT,
            "transcript": transcript,
            "normalized_wer": normalized_wer,
            "max_normalized_wer": 0.85,
            "quality_bearing": True,
            "output_dir": str(output_dir),
            "score_status": scored.get("status"),
            "hf_whisper_safetensors_score": {
                "candidate_id": run_id,
                "normalized_wer": score.get("normalized_wer"),
                "balanced_score": score.get("balanced_score"),
                "balanced_rank": score.get("balanced_rank"),
                "alignment_mode": score.get("alignment_mode"),
            },
            "dependency_versions": package_versions(["torch", "transformers", "safetensors", "sentencepiece", "tokenizers", "torchaudio", "llama-cpp-python"]),
            **dependency_report_details,
            "failures": failures,
            **scan_details,
        },
        artifacts=[
            SMOLLM_PATH,
            *model_dir.glob("*.safetensors"),
            source,
            output_dir / "results.json",
            output_dir / "results.txt",
            output_dir / "benchmark.csv",
            output_dir / "compare.html",
            scored_path,
            scored_html,
        ],
    )


def _run_ctc_quality(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    dependency_block = _ensure_transformers_deps(row_id, evidence_dir, install_deps, [SMOLLM_PATH])
    if dependency_block is not None:
        return dependency_block
    model_dir_or_row = _ensure_repo(row_id, evidence_dir, HF_CTC_QUALITY_REPO, allow_downloads, [SMOLLM_PATH])
    if isinstance(model_dir_or_row, dict):
        return model_dir_or_row
    model_dir = model_dir_or_row
    config = smoke_config(evidence_dir, "cpu")
    config["runtime"]["llm_context_tokens"] = 1024
    config["runtime"]["llm_reference_max_tokens"] = 128
    config["runtime"]["llm_reference_temperature"] = 0.0
    runnable, unsupported = scan_models(model_dir)
    candidates = [candidate for candidate in runnable if candidate.adapter_name == "hf_transformers_asr"]
    if not candidates:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary=f"{HF_CTC_QUALITY_REPO} did not scan as a runnable HF non-Whisper Safetensors ASR model.",
            details={
                "repo_id": HF_CTC_QUALITY_REPO,
                "runnable": [candidate.adapter_name for candidate in runnable],
                "unsupported": [{"adapter_name": candidate.adapter_name, "missing": candidate.missing_files, "warnings": candidate.warnings} for candidate in unsupported],
            },
            artifacts=[SMOLLM_PATH, *model_dir.glob("*.safetensors")],
        )
    llm_candidate, scan_details = _smollm_candidate()
    if llm_candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference/correction LLM candidate.",
            details=scan_details,
            artifacts=[SMOLLM_PATH, *model_dir.glob("*.safetensors")],
        )
    source = Path(config["folders"]["input"]) / "hf_safetensors_ctc_quality_sapi.wav"
    generate_windows_sapi_wav(source, REFERENCE_TEXT)
    output_dir = process_file_with_candidates(source, [candidates[0]], config, unsupported, reference_llm=llm_candidate)
    if output_dir is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="HF non-Whisper Safetensors quality row did not produce a report directory.",
            details={"repo_id": HF_CTC_QUALITY_REPO, **scan_details},
            artifacts=[SMOLLM_PATH, *model_dir.glob("*.safetensors"), source],
        )
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
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

    dependency_report_failures, dependency_report_details = dependency_resolution_report_failures(results, expected_groups={"transformers_cpu", "llama_cpp"})
    failures: list[str] = list(dependency_report_failures)
    if not transcript.strip():
        failures.append("HF non-Whisper Safetensors speech transcript was empty")
    if normalized_wer > 0.85:
        failures.append(f"HF non-Whisper Safetensors normalized WER {normalized_wer:.3f} exceeded threshold 0.850")
    for name in ["results.json", "results.txt", "benchmark.csv", "compare.html", "scored_report.json", "compare_scored.html"]:
        if not (output_dir / name).exists():
            failures.append(f"missing report artifact {name}")
    scored_html_text = scored_html.read_text(encoding="utf-8")
    if "Loaded precomputed LLM-corrected reference scores" not in scored_html_text:
        failures.append("compare_scored.html missing precomputed score marker")
    run_id = runs[0]["model"]["candidate_id"] if runs else ""
    score = scored.get("scores", {}).get(run_id, {})
    if scored.get("status") != "scored" or score.get("normalized_wer") is None:
        failures.append("HF non-Whisper Safetensors scored reference was not produced")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "HF non-Whisper Safetensors transcribed generated speech with acceptable WER, then SmolLM scoring/report validation completed."
            if not failures
            else "HF non-Whisper Safetensors quality SmolLM grading validation failed."
        ),
        details={
            "repo_id": HF_CTC_QUALITY_REPO,
            "adapter_name": "hf_transformers_asr",
            "reference_text": REFERENCE_TEXT,
            "transcript": transcript,
            "normalized_wer": normalized_wer,
            "max_normalized_wer": 0.85,
            "quality_bearing": True,
            "output_dir": str(output_dir),
            "score_status": scored.get("status"),
            "hf_safetensors_ctc_score": {
                "candidate_id": run_id,
                "normalized_wer": score.get("normalized_wer"),
                "balanced_score": score.get("balanced_score"),
                "balanced_rank": score.get("balanced_rank"),
                "alignment_mode": score.get("alignment_mode"),
            },
            "dependency_versions": package_versions(["torch", "transformers", "safetensors", "sentencepiece", "tokenizers", "torchaudio", "llama-cpp-python"]),
            **dependency_report_details,
            "failures": failures,
            **scan_details,
        },
        artifacts=[
            SMOLLM_PATH,
            *model_dir.glob("*.safetensors"),
            source,
            output_dir / "results.json",
            output_dir / "results.txt",
            output_dir / "benchmark.csv",
            output_dir / "compare.html",
            scored_path,
            scored_html,
        ],
    )


def run(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    if row_id not in {
        "hf_whisper_safetensors_smollm_grading_cpu",
        "hf_whisper_sharded_safetensors_smollm_grading_cpu",
        "hf_safetensors_asr_smollm_grading_cpu",
        "hf_whisper_safetensors_quality_smollm_grading_cpu",
        "hf_safetensors_asr_quality_smollm_grading_cpu",
    }:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary=f"Unsupported HF Safetensors SmolLM grading row id: {row_id}",
            details={"row_id": row_id},
        )
    if not SMOLLM_PATH.exists():
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="SmolLM 135M GGUF fixture is not present locally, so HF Safetensors output cannot be graded by the local reference path.",
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
            summary="llama-cpp-python is not installed, so SmolLM GGUF cannot run after HF Safetensors ASR.",
            block_reason="missing llama_cpp import",
            external_requirement="install llama_cpp dependency group",
            details={"dependency_versions": package_versions(["llama-cpp-python"])},
            artifacts=[SMOLLM_PATH],
        )
    if row_id == "hf_whisper_safetensors_quality_smollm_grading_cpu":
        return _run_whisper_quality(row_id, evidence_dir, install_deps, allow_downloads)
    if row_id == "hf_safetensors_asr_quality_smollm_grading_cpu":
        return _run_ctc_quality(row_id, evidence_dir, install_deps, allow_downloads)
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import safetensors  # noqa: F401
    except ModuleNotFoundError as exc:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Transformers/Safetensors dependency group is not installed, so HF Safetensors plus SmolLM grading cannot run.",
            block_reason=f"missing {exc.name}",
            external_requirement="python -m pip install -r requirements/transformers_cpu.txt",
            details={"dependency_versions": package_versions(["torch", "transformers", "safetensors", "sentencepiece", "tokenizers", "torchaudio"])},
            artifacts=[SMOLLM_PATH],
        )

    base_row = "hf_whisper_safetensors_cpu" if "whisper" in row_id else "hf_safetensors_asr"
    repo_id, adapter_name = _repo_for_row(base_row)
    model_dir = evidence_dir / "Models" / repo_id.replace("/", "__")
    if not list(model_dir.glob("*.safetensors")):
        if not allow_downloads:
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary=f"{repo_id} Safetensors fixture is not cached locally.",
                block_reason=f"missing local fixture {model_dir}",
                external_requirement=f"rerun with --allow-downloads to download {repo_id}",
                details={"repo_id": repo_id, "adapter_name": adapter_name},
                artifacts=[SMOLLM_PATH],
            )
        try:
            _download_repo(repo_id, model_dir)
        except Exception as exc:
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary=f"Could not download {repo_id} Safetensors fixture.",
                block_reason=f"{type(exc).__name__}: {exc}",
                external_requirement=f"network access to https://huggingface.co/{repo_id}",
                details={"repo_id": repo_id, "adapter_name": adapter_name},
                artifacts=[SMOLLM_PATH],
            )

    sharded_artifacts: list[Path] = []
    if row_id == "hf_whisper_sharded_safetensors_smollm_grading_cpu":
        try:
            sharded_dir = evidence_dir / "Models" / "tiny-random-whisper-sharded-safetensors"
            sharded_artifacts = _materialize_sharded_whisper(model_dir, sharded_dir)
            model_dir = sharded_dir
        except Exception as exc:
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary="Could not materialize a complete sharded Whisper Safetensors fixture.",
                block_reason=f"{type(exc).__name__}: {exc}",
                external_requirement=f"download and load {HF_WHISPER_SHARDED_REPO}, then save_pretrained(..., safe_serialization=True, max_shard_size='50KB')",
                details={"repo_id": HF_WHISPER_SHARDED_REPO, "dependency_versions": package_versions(["torch", "transformers", "safetensors"])},
                artifacts=[SMOLLM_PATH, *model_dir.glob("*.safetensors")],
            )
        if len([path for path in sharded_artifacts if path.suffix == ".safetensors"]) < 2 or not any(".safetensors.index" in path.name for path in sharded_artifacts):
            return write_row(
                row_id,
                "fail",
                evidence_dir,
                summary="Sharded Whisper Safetensors fixture did not produce multiple shards plus an index.",
                details={"sharded_artifacts": [str(path) for path in sharded_artifacts]},
                artifacts=[SMOLLM_PATH, *sharded_artifacts],
            )

    runnable, unsupported = scan_models(model_dir)
    candidates = [candidate for candidate in runnable if candidate.adapter_name == adapter_name]
    if not candidates:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary=f"{repo_id} did not scan as a runnable {adapter_name} Safetensors model.",
            details={
                "repo_id": repo_id,
                "adapter_name": adapter_name,
                "runnable": [candidate.adapter_name for candidate in runnable],
                "unsupported": [{"adapter_name": candidate.adapter_name, "missing": candidate.missing_files, "warnings": candidate.warnings} for candidate in unsupported],
            },
            artifacts=[SMOLLM_PATH, *model_dir.glob("*.safetensors")],
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
            summary=f"{repo_id} Safetensors fixture could not load or run through Transformers.",
            block_reason=f"{type(exc).__name__}: {exc}",
            external_requirement="repair transformers_cpu dependency group or use a different complete HF ASR Safetensors fixture",
            details={"repo_id": repo_id, "adapter_name": adapter_name, "dependency_versions": package_versions(["torch", "transformers", "safetensors", "sentencepiece", "tokenizers", "torchaudio"])},
            artifacts=[SMOLLM_PATH, *model_dir.glob("*.safetensors")],
        )
    transcript = result.transcript_chunks[0].text if result.transcript_chunks else ""
    if "whisper" in row_id and not transcript.strip():
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="HF Whisper Safetensors tiny fixture produced an empty transcript before SmolLM grading.",
            details={"repo_id": repo_id, "adapter_name": adapter_name, "metrics": result.metrics, "errors": result.errors},
            artifacts=[SMOLLM_PATH, *model_dir.glob("*.safetensors")],
        )

    llm_candidate, scan_details = _smollm_candidate()
    if llm_candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference/correction LLM candidate.",
            details=scan_details,
            artifacts=[SMOLLM_PATH, *model_dir.glob("*.safetensors")],
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
            summary="SmolLM GGUF loaded after HF Safetensors ASR but generated empty text.",
            details={"dependency_versions": package_versions(["llama-cpp-python"])},
            artifacts=[SMOLLM_PATH, *model_dir.glob("*.safetensors")],
        )

    source = evidence_dir / "Input" / f"{row_id}.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"easy-asr-bench-hf-safetensors-smollm-grading")
    chunks = [SimpleNamespace(index=0, start_seconds=0.0, end_seconds=1.0, cut_reason="safetensors_fixture", rms_db=-20.0)]
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
        "note": "This row proves SmolLM runs after HF Safetensors output. Stable scoring uses the tiny fixture output as structural reference.",
    }
    output_dir = write_all_reports(results, evidence_dir / "Output")
    reference = _reference_for(results, transcript)
    scored = import_llm_reference(results, "SmolLM corrected reference fixture:\n```json\n" + json.dumps(reference) + "\n```")
    scored_path = output_dir / "scored_report.json"
    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    scored_results = dict(results)
    if scored.get("status") == "scored":
        scored_results["reference_scores"] = scored["scores"]
    scored_html = output_dir / "compare_scored.html"
    scored_html.write_text(build_html_report(scored_results), encoding="utf-8", newline="\n")

    dependency_report_failures, dependency_report_details = dependency_resolution_report_failures(results, expected_groups={"transformers_cpu", "llama_cpp"})
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
        failures.append("HF Safetensors scored reference was not produced")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            f"{repo_id} Safetensors output was followed by SmolLM GGUF generation and scored report validation."
            if not failures
            else "HF Safetensors SmolLM grading validation failed."
        ),
        details={
            "repo_id": repo_id,
            "adapter_name": adapter_name,
            "transcript": transcript,
            "quality_bearing": False,
            "quality_note": "Tiny Safetensors fixture is used for structural backend/report regression only; real WER proof remains pending.",
            "sharded_safetensors": row_id == "hf_whisper_sharded_safetensors_smollm_grading_cpu",
            "sharded_artifacts": [str(path) for path in sharded_artifacts],
            "metrics": result.metrics,
            "errors": result.errors,
            "output_dir": str(output_dir),
            "candidate_id": llm_candidate.candidate_id,
            "generated_text": generated_text,
            "score_status": scored.get("status"),
            "hf_safetensors_score": {
                "candidate_id": run_id,
                "normalized_wer": score.get("normalized_wer"),
                "balanced_score": score.get("balanced_score"),
                "balanced_rank": score.get("balanced_rank"),
                "alignment_mode": score.get("alignment_mode"),
            },
            "dependency_versions": package_versions(["torch", "transformers", "safetensors", "sentencepiece", "tokenizers", "torchaudio", "llama-cpp-python"]),
            **dependency_report_details,
            "failures": failures,
            **scan_details,
        },
        artifacts=[
            SMOLLM_PATH,
            *model_dir.glob("*.safetensors"),
            *model_dir.glob("*.safetensors.index*.json"),
            output_dir / "results.json",
            output_dir / "results.txt",
            output_dir / "benchmark.csv",
            output_dir / "compare.html",
            scored_path,
            scored_html,
        ],
    )
