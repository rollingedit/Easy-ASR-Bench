from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import sys
import re
from datetime import datetime
from itertools import combinations
from pathlib import Path

from .html_report_builder import build_html_report
from . import __version__
from .version import RELEASE_CHANNEL, RELEASE_COMMIT
from .dependency_manager import cuda_diagnostics, onnxruntime_available_providers
from .llm_reference_prompt import build_llm_reference_prompt
from .results_schema import validate_results_schema
from .reference_scoring import runtime_rankings
from .scoring import pairwise_metrics
from .utils import format_timestamp, now_stamp, safe_stem, sha256_file


DEPENDENCY_PACKAGES = [
    "imageio-ffmpeg",
    "librosa",
    "numpy",
    "onnxruntime",
    "psutil",
    "soundfile",
    "torch",
    "transformers",
    "faster-whisper",
    "ctranslate2",
    "pywhispercpp",
    "openai-whisper",
    "llama-cpp-python",
]


def _default_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def format_error_summary(error) -> str:
    if not isinstance(error, dict):
        return str(error)
    stage = error.get("stage", "unknown")
    message = error.get("message", "")
    model_name = error.get("model_name") or error.get("model_id") or ""
    prefix = f"{model_name} | " if model_name else ""
    return f"{prefix}{stage}: {message}".strip()


def normalize_run_error(error, candidate) -> dict | str:
    if isinstance(error, dict):
        return error
    text = str(error)
    match = re.match(r"^(?P<chunk_id>(?:chunk-)?\d{1,5}):\s*(?P<message>.+)$", text, re.IGNORECASE)
    if not match:
        return text
    chunk_id = match.group("chunk_id")
    message = match.group("message")
    return {
        "status": "chunk_failed",
        "stage": "chunk_inference",
        "chunk_id": chunk_id,
        "model_id": candidate.candidate_id,
        "model_name": candidate.display_name,
        "model_path": str(candidate.path),
        "error_type": "ChunkInferenceError",
        "message": message,
        "likely_causes": [
            "This chunk failed during model inference while other chunks or models may still be usable.",
            "The selected model/backend may have hit a provider, memory, decode, or input-shape issue for this chunk.",
        ],
        "next_actions": [
            "Open the detailed log and inspect this chunk id.",
            "Retry with CPU or a smaller model if the message mentions CUDA, GPU, memory, or provider failure.",
            "If only this chunk failed, inspect the source audio around the chunk timestamp for noise, silence, or corruption.",
        ],
        "dependency_group": ", ".join(candidate.dependency_groups),
        "provider_requested": "",
        "provider_actual": "",
        "repair_command": "Run setup.bat --doctor for runtime status and repair commands.",
        "log_path": "",
        "traceback": "",
    }


def candidate_to_dict(candidate) -> dict:
    runtime_precision_supported = bool(candidate.runnable)
    runtime_precision_reason = candidate.metadata.get("runtime_precision_reason", "")
    if not runtime_precision_supported and not runtime_precision_reason:
        runtime_precision_reason = "Detected precision is reported for identification; runtime support depends on the selected adapter and complete required files."
    return {
        "candidate_id": candidate.candidate_id,
        "display_name": candidate.display_name,
        "family_name": candidate.family_name,
        "backend": candidate.backend,
        "container_format": candidate.container_format,
        "task": candidate.task,
        "precision": candidate.precision,
        "detected_precision": candidate.precision,
        "precision_bucket": candidate.quantization_label,
        "runtime_precision_supported": runtime_precision_supported,
        "runtime_precision_reason": runtime_precision_reason,
        "path": str(candidate.path),
        "adapter_name": candidate.adapter_name,
        "warnings": list(candidate.warnings),
    }


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in DEPENDENCY_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _compact_runtime_resolution(path: Path, resolution: dict) -> dict:
    return {
        "path": str(path),
        "dependency_group": resolution.get("dependency_group", ""),
        "status": resolution.get("status", ""),
        "backend_verified": bool(resolution.get("backend_verified", False)),
        "backend_probe_kind": resolution.get("backend_probe_kind", ""),
        "accelerator_requested": bool(resolution.get("accelerator_requested", False)),
        "accelerator": resolution.get("accelerator", ""),
        "accelerator_verified": bool(resolution.get("accelerator_verified", False)),
        "versions": dict(resolution.get("versions") or {}),
        "providers": list(resolution.get("providers") or []),
        "runtime_path": str(resolution.get("runtime_path") or ""),
        "config_runtime": dict(resolution.get("config_runtime") or {}),
    }


def _compact_repair_record(record: dict) -> dict:
    after = record.get("after", {})
    backend_probe = after.get("backend_probe", {})
    accelerator_probe = backend_probe.get("accelerator_probe", {})
    return {
        "dependency_group": record.get("affected_dependency_group", ""),
        "status": record.get("status", ""),
        "repair_result": after.get("repair_result", ""),
        "backend_probe_kind": backend_probe.get("kind", ""),
        "cached_runtime_resolution_status": record.get("cached_runtime_resolution_check", {}).get("status", ""),
        "previous_runtime_resolution_status": after.get("previous_runtime_resolution_check", {}).get("status", ""),
        "runtime_resolution_path": after.get("runtime_resolution_path", ""),
        "accelerator_requested": bool(accelerator_probe.get("requested", False)),
        "accelerator": accelerator_probe.get("accelerator", ""),
        "accelerator_verified": bool(accelerator_probe.get("ok", False)),
    }


def dependency_resolution_environment(project_root: Path | None = None) -> dict:
    project_root = project_root or _default_project_root()
    logs_dir = project_root / "Logs"
    resolutions = []
    invalid_resolution_files = []
    if logs_dir.exists():
        for path in sorted(logs_dir.glob("dependency_resolution_*.json")):
            try:
                resolution = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                invalid_resolution_files.append({"path": str(path), "error": str(exc)})
                continue
            if resolution.get("schema") != "easy_asr_bench.runtime_resolution.v1":
                invalid_resolution_files.append({"path": str(path), "error": "unexpected runtime resolution schema"})
                continue
            resolutions.append(_compact_runtime_resolution(path, resolution))
    summary = {
        "schema": "easy_asr_bench.dependency_resolution_environment.v1",
        "resolution_count": len(resolutions),
        "invalid_resolution_files": len(invalid_resolution_files),
        "backend_verified": sum(1 for item in resolutions if item["backend_verified"]),
        "accelerator_requested": sum(1 for item in resolutions if item["accelerator_requested"]),
        "accelerator_unverified": sum(1 for item in resolutions if item["accelerator_requested"] and not item["accelerator_verified"]),
    }
    environment = {
        "summary": summary,
        "resolutions": resolutions,
        "invalid_resolution_files": invalid_resolution_files,
    }
    repair_path = logs_dir / "repair_all_safe_last.json"
    if repair_path.exists():
        try:
            repair = json.loads(repair_path.read_text(encoding="utf-8"))
            environment["last_repair_all_safe"] = {
                "path": str(repair_path),
                "mode": repair.get("mode", ""),
                "summary": dict(repair.get("summary") or {}),
                "records": [_compact_repair_record(record) for record in repair.get("records", [])],
            }
        except (OSError, json.JSONDecodeError) as exc:
            environment["last_repair_all_safe"] = {"path": str(repair_path), "error": str(exc)}
    return environment


def runtime_environment(project_root: Path | None = None) -> dict:
    environment = {
        "app_version": __version__,
        "release_channel": RELEASE_CHANNEL,
        "release_commit": RELEASE_COMMIT,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "cuda_available": False,
        "gpu": [],
    }
    try:
        import torch

        environment["cuda_available"] = bool(torch.cuda.is_available())
        environment["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            environment["gpu"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except Exception:
        pass
    providers, provider_error = onnxruntime_available_providers()
    environment["onnxruntime_providers"] = providers
    if provider_error:
        environment["onnxruntime_provider_probe_error"] = provider_error
    environment["cuda_diagnostics"] = cuda_diagnostics()
    environment["dependency_resolution_environment"] = dependency_resolution_environment(project_root)
    return environment


def build_failed_file_results(
    *,
    source_path: Path,
    stage: str,
    error_type: str,
    message: str,
    likely_causes: list[str],
    next_actions: list[str],
    log_path: Path | None,
    selected_models: list | None = None,
    unsupported_models: list | None = None,
) -> dict:
    error = {
        "status": "failed_before_model_run",
        "stage": stage,
        "error_type": error_type,
        "message": message,
        "likely_causes": likely_causes,
        "next_actions": next_actions,
        "log_path": str(log_path) if log_path else "",
        "source_files_modified": False,
    }
    results = {
        "schema": "easy_asr_bench.results.v1",
        "app_version": __version__,
        "created_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "path": str(source_path),
            "name": source_path.name,
            "sha256": "",
            "duration_seconds": 0,
        },
        "environment": runtime_environment(),
        "dependency_versions": dependency_versions(),
        "adapter_versions": {},
        "chunk_plan": {"sample_rate": 16000, "source_audio_seconds": 0, "chunks": []},
        "selected_models": [candidate_to_dict(candidate) for candidate in selected_models or []],
        "runs": [],
        "unsupported_models": [
            candidate_to_dict(candidate) | {"missing_files": candidate.missing_files, "help_text": candidate.help_text}
            for candidate in unsupported_models or []
        ],
        "pairwise_differences": {},
        "errors": [error],
    }
    schema_errors = validate_results_schema(results)
    if schema_errors:
        raise ValueError("Invalid failed-file results schema: " + "; ".join(schema_errors))
    return results


def build_results(
    source_path: Path,
    audio_seconds: float,
    chunks: list,
    run_results: list,
    unsupported_models: list,
    media_seconds: float,
    errors: list[str] | None = None,
) -> dict:
    source_hash = ""
    try:
        source_hash = sha256_file(source_path)
    except OSError:
        pass
    chunk_plan = {
        "sample_rate": 16000,
        "source_audio_seconds": audio_seconds,
        "chunks": [
            {
                "chunk_id": f"{chunk.index + 1:04d}",
                "index": chunk.index,
                "start_seconds": chunk.start_seconds,
                "end_seconds": chunk.end_seconds,
                "start_timestamp": format_timestamp(chunk.start_seconds),
                "end_timestamp": format_timestamp(chunk.end_seconds),
                "cut_reason": getattr(chunk, "cut_reason", "rms_or_duration"),
                "rms_db": getattr(chunk, "rms_db", None),
            }
            for chunk in chunks
        ],
    }
    runs = []
    for result in run_results:
        candidate = result.candidate
        metrics = dict(result.metrics)
        metrics.setdefault("provider", candidate.backend)
        metrics.setdefault("media_normalization_seconds", media_seconds)
        metrics.setdefault("audio_seconds_per_wall_second", metrics.get("audio_seconds", 0) / max(0.001, metrics.get("total_wall_seconds", 0.001)))
        metrics.setdefault("peak_vram_mb", None)
        metrics.setdefault("vram_measurement_source", "unavailable")
        metrics.setdefault("torch_peak_vram_mb", None)
        metrics.setdefault("windows_peak_dedicated_vram_mb", None)
        metrics.setdefault(
            "vram_measurement_note",
            "VRAM telemetry was unavailable or not reported by this backend/run.",
        )
        normalized_errors = [normalize_run_error(error, candidate) for error in result.errors]
        runs.append(
            {
                "model": candidate_to_dict(candidate),
                "transcript_chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "start_seconds": chunk.start_seconds,
                        "end_seconds": chunk.end_seconds,
                        "text": chunk.text,
                        "raw": chunk.raw,
                    }
                    for chunk in result.transcript_chunks
                ],
                "metrics": metrics,
                "errors": normalized_errors,
            }
        )
    pairwise = {}
    for left, right in combinations(runs, 2):
        left_text = "\n".join(chunk["text"] for chunk in left["transcript_chunks"])
        right_text = "\n".join(chunk["text"] for chunk in right["transcript_chunks"])
        pairwise[f"{left['model']['candidate_id']}__vs__{right['model']['candidate_id']}"] = pairwise_metrics(left_text, right_text)
    results = {
        "schema": "easy_asr_bench.results.v1",
        "app_version": __version__,
        "created_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "path": str(source_path),
            "name": source_path.name,
            "sha256": source_hash,
            "duration_seconds": audio_seconds,
        },
        "environment": runtime_environment(),
        "dependency_versions": dependency_versions(),
        "adapter_versions": {run["model"]["adapter_name"]: __version__ for run in runs},
        "chunk_plan": chunk_plan,
        "selected_models": [run["model"] for run in runs],
        "runs": runs,
        "unsupported_models": [candidate_to_dict(candidate) | {"missing_files": candidate.missing_files, "help_text": candidate.help_text} for candidate in unsupported_models],
        "pairwise_differences": pairwise,
        "runtime_rankings": runtime_rankings({"runs": runs}),
        "errors": errors or [],
    }
    schema_errors = validate_results_schema(results)
    if schema_errors:
        raise ValueError("Invalid results schema: " + "; ".join(schema_errors))
    return results


def write_all_reports(results: dict, output_root: Path) -> Path:
    source_name = safe_stem(Path(results["source"]["name"]))
    output_dir = output_root / f"{source_name}__{now_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    txt_path = output_dir / "results.txt"
    html_path = output_dir / "compare.html"
    csv_path = output_dir / "benchmark.csv"

    _atomic_write_text(json_path, json.dumps(results, ensure_ascii=False, indent=2))
    _atomic_write_text(txt_path, render_text_report(results))
    _atomic_write_text(html_path, build_html_report(results))
    write_benchmark_csv(csv_path, results)
    write_prompt_packs(output_dir, results)
    return output_dir


def render_text_report(results: dict) -> str:
    if results.get("errors") and not results.get("runs"):
        lines = [
            "Easy ASR Bench Failed File Report",
            "=" * 80,
            f"Created: {results['created_local']}",
            f"Source: {results['source']['path']}",
            "",
            "This file could not be processed before model benchmarking started.",
            "",
        ]
        for error in results["errors"]:
            if isinstance(error, dict):
                lines.extend(
                    [
                        f"Stage: {error.get('stage', 'unknown')}",
                        f"Problem: {error.get('message', '')}",
                        "",
                        "Likely causes:",
                    ]
                )
                lines.extend(f"- {item}" for item in error.get("likely_causes", []))
                lines.extend(["", "Next actions:"])
                lines.extend(f"- {item}" for item in error.get("next_actions", []))
                if error.get("log_path"):
                    lines.extend(["", f"Detailed log: {error['log_path']}"])
                lines.extend(["", "No source files were modified."])
            else:
                lines.append(str(error))
        return "\n".join(lines) + "\n"
    lines = [
        "Easy ASR Bench Results",
        "=" * 80,
        f"Created: {results['created_local']}",
        f"Source: {results['source']['path']}",
        f"Duration: {format_timestamp(float(results['source']['duration_seconds']))}",
        f"Chunks: {len(results['chunk_plan']['chunks'])}",
        "Timestamp note: timestamps are chunk timestamps, not word-level timestamps.",
        "",
        "Models Tested",
        "-" * 80,
    ]
    for run in results["runs"]:
        metrics = run["metrics"]
        lines.append(
            f"{run['model']['display_name']} | {run['model']['precision']} | "
            f"{metrics.get('audio_seconds_per_wall_second', 0):.3f}x realtime | "
            f"{metrics.get('peak_process_memory_mb', 0):.0f} MB RAM | "
            f"VRAM: {_format_optional_mb(metrics.get('peak_vram_mb'))} ({metrics.get('vram_measurement_source', 'unavailable')}) | "
            f"errors: {len(run.get('errors', []))}"
        )
    lines.extend(["", "Transcripts", "-" * 80])
    for run in results["runs"]:
        lines.extend(["", f"MODEL: {run['model']['display_name']} ({run['model']['precision']})", "~" * 80])
        for chunk in run["transcript_chunks"]:
            lines.append(f"[{format_timestamp(chunk['start_seconds'])} - {format_timestamp(chunk['end_seconds'])}] {chunk['text']}")
        if run["errors"]:
            lines.append("Errors:")
            for error in run["errors"]:
                lines.append(f"- {format_error_summary(error)}")
                if isinstance(error, dict):
                    causes = error.get("likely_causes", [])
                    actions = error.get("next_actions", [])
                    if causes:
                        lines.append("  Likely causes:")
                        lines.extend(f"  - {item}" for item in causes)
                    if actions:
                        lines.append("  Next actions:")
                        lines.extend(f"  - {item}" for item in actions)
                    if error.get("repair_command"):
                        lines.append(f"  Repair command: {error['repair_command']}")
                    if error.get("log_path"):
                        lines.append(f"  Detailed log: {error['log_path']}")
    lines.extend(["", "Pairwise Differences", "-" * 80])
    for name, metrics in results["pairwise_differences"].items():
        lines.append(f"{name}: normalized WER-like difference {metrics['normalized_wer_like_difference']:.4f}, CER difference {metrics['cer_difference']:.4f}")
    if results.get("runtime_rankings", {}).get("rows"):
        lines.extend(["", "Runtime-Only Ranking", "-" * 80])
        lines.append("This ranking compares speed and memory only; it does not measure transcript quality.")
        for row in results["runtime_rankings"]["rows"]:
            lines.append(
                f"#{row['runtime_rank']} {row['display_name']} | "
                f"speed percentile: {_format_optional_number(row.get('speed_percentile'))} | "
                f"memory percentile: {_format_optional_number(row.get('memory_percentile_inverse'))}"
            )
    if results.get("reference_llm"):
        lines.extend(["", "Local GGUF Reference/Correction LLM", "-" * 80])
        lines.append(f"Selected: {results['reference_llm']['display_name']}")
        attempt = results.get("local_llm_reference_attempt")
        if attempt:
            lines.append(f"Status: {attempt.get('status')}")
            if attempt.get("error"):
                lines.append(f"Error: {attempt['error']}")
            if attempt.get("raw_response"):
                lines.extend(["", "Raw local LLM response:", attempt["raw_response"]])
    lines.extend(["", "LLM-Corrected Reference Instructions", "-" * 80, build_llm_reference_prompt(results)])
    return "\n".join(lines) + "\n"


def write_benchmark_csv(path: Path, results: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_id",
                "display_name",
                "backend",
                "precision",
                "audio_seconds",
                "chunk_count",
                "model_load_seconds",
                "inference_seconds",
                "total_wall_seconds",
                "audio_seconds_per_wall_second",
                "tokens_generated",
                "peak_process_memory_mb",
                "peak_vram_mb",
                "vram_measurement_source",
                "torch_peak_vram_mb",
                "windows_peak_dedicated_vram_mb",
                "errors",
            ],
        )
        writer.writeheader()
        for run in results["runs"]:
            metrics = run["metrics"]
            writer.writerow(
                {
                    "model_id": run["model"]["candidate_id"],
                    "display_name": run["model"]["display_name"],
                    "backend": run["model"]["backend"],
                    "precision": run["model"]["precision"],
                    "audio_seconds": results["source"]["duration_seconds"],
                    "chunk_count": len(results["chunk_plan"]["chunks"]),
                    "model_load_seconds": metrics.get("model_load_seconds", 0),
                    "inference_seconds": metrics.get("inference_seconds", 0),
                    "total_wall_seconds": metrics.get("total_wall_seconds", 0),
                    "audio_seconds_per_wall_second": metrics.get("audio_seconds_per_wall_second", 0),
                    "tokens_generated": metrics.get("tokens_generated", 0),
                    "peak_process_memory_mb": metrics.get("peak_process_memory_mb", 0),
                    "peak_vram_mb": metrics.get("peak_vram_mb"),
                    "vram_measurement_source": metrics.get("vram_measurement_source", "unavailable"),
                    "torch_peak_vram_mb": metrics.get("torch_peak_vram_mb"),
                    "windows_peak_dedicated_vram_mb": metrics.get("windows_peak_dedicated_vram_mb"),
                    "errors": len(run.get("errors", [])),
                }
            )
    partial.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def _format_optional_mb(value) -> str:
    if value is None:
        return "unavailable"
    try:
        return f"{float(value):.0f} MB"
    except (TypeError, ValueError):
        return "unavailable"


def _format_optional_number(value) -> str:
    if value is None:
        return "unavailable"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "unavailable"


def write_prompt_packs(output_dir: Path, results: dict, max_chars: int = 24000) -> None:
    prompt = build_llm_reference_prompt(results)
    if len(prompt) <= max_chars:
        _atomic_write_text(output_dir / "results_llm_prompt_part_001.txt", prompt)
        return
    chunks = results.get("chunk_plan", {}).get("chunks", [])
    for index, chunk in enumerate(chunks, 1):
        part = dict(results)
        wanted = chunk["chunk_id"]
        part["chunk_plan"] = dict(results["chunk_plan"])
        part["chunk_plan"]["chunks"] = [chunk]
        part["runs"] = []
        for run in results.get("runs", []):
            run_part = dict(run)
            run_part["transcript_chunks"] = [item for item in run.get("transcript_chunks", []) if item.get("chunk_id") == wanted]
            part["runs"].append(run_part)
        _atomic_write_text(output_dir / f"results_llm_prompt_part_{index:03d}.txt", build_llm_reference_prompt(part))
