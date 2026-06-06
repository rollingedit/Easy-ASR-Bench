from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

from .html_report_builder import build_html_report
from . import __version__
from .dependency_manager import cuda_diagnostics
from .llm_reference_prompt import build_llm_reference_prompt
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


def candidate_to_dict(candidate) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "display_name": candidate.display_name,
        "family_name": candidate.family_name,
        "backend": candidate.backend,
        "container_format": candidate.container_format,
        "task": candidate.task,
        "precision": candidate.precision,
        "precision_bucket": candidate.quantization_label,
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


def runtime_environment() -> dict:
    environment = {
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
    try:
        import onnxruntime as ort

        environment["onnxruntime_providers"] = ort.get_available_providers()
    except Exception:
        environment["onnxruntime_providers"] = []
    environment["cuda_diagnostics"] = cuda_diagnostics()
    return environment


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
        metrics.setdefault("media_normalization_seconds", media_seconds)
        metrics.setdefault("audio_seconds_per_wall_second", metrics.get("audio_seconds", 0) / max(0.001, metrics.get("total_wall_seconds", 0.001)))
        metrics.setdefault("peak_vram_mb", None)
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
                "errors": result.errors,
            }
        )
    pairwise = {}
    for left, right in combinations(runs, 2):
        left_text = "\n".join(chunk["text"] for chunk in left["transcript_chunks"])
        right_text = "\n".join(chunk["text"] for chunk in right["transcript_chunks"])
        pairwise[f"{left['model']['candidate_id']}__vs__{right['model']['candidate_id']}"] = pairwise_metrics(left_text, right_text)
    return {
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
        "errors": errors or [],
    }


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
            f"{metrics.get('peak_process_memory_mb', 0):.0f} MB RAM | errors: {len(run.get('errors', []))}"
        )
    lines.extend(["", "Transcripts", "-" * 80])
    for run in results["runs"]:
        lines.extend(["", f"MODEL: {run['model']['display_name']} ({run['model']['precision']})", "~" * 80])
        for chunk in run["transcript_chunks"]:
            lines.append(f"[{format_timestamp(chunk['start_seconds'])} - {format_timestamp(chunk['end_seconds'])}] {chunk['text']}")
        if run["errors"]:
            lines.append("Errors:")
            lines.extend(f"- {error}" for error in run["errors"])
    lines.extend(["", "Pairwise Differences", "-" * 80])
    for name, metrics in results["pairwise_differences"].items():
        lines.append(f"{name}: normalized WER-like difference {metrics['normalized_wer_like_difference']:.4f}, CER difference {metrics['cer_difference']:.4f}")
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
                    "errors": len(run.get("errors", [])),
                }
            )
    partial.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


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
