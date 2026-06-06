from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from .benchmark import VariantMetrics
from .text_compare import compare_texts
from .utils import format_timestamp, now_stamp, safe_stem


def render_variant_section(variant: str, chunks: list[dict], metrics: VariantMetrics, error: str | None = None) -> str:
    lines = [
        "=" * 80,
        f"VARIANT: {variant}",
        f"Precision: {metrics.precision}",
        f"Provider: {metrics.provider}",
        "=" * 80,
        "",
    ]
    if error:
        lines.extend(["ERROR:", error, ""])
    else:
        for chunk in chunks:
            lines.append(
                f"[{format_timestamp(chunk['start_seconds'])} - {format_timestamp(chunk['end_seconds'])}] "
                f"{chunk['text'].strip()}"
            )
        lines.append("")
    lines.extend(
        [
            "Performance:",
            f"  model_load_seconds: {metrics.model_load_seconds:.3f}",
            f"  inference_seconds: {metrics.inference_seconds:.3f}",
            f"  total_wall_seconds: {metrics.total_wall_seconds:.3f}",
            f"  audio_seconds_per_wall_second: {metrics.audio_seconds_per_wall_second:.3f}x",
            f"  wall_seconds_per_audio_second: {metrics.real_time_factor:.3f}",
            f"  tokens_generated: {metrics.tokens_generated}",
            f"  tokens_per_second: {metrics.tokens_per_second:.3f}",
            f"  peak_ram_mb: {metrics.peak_process_memory_mb:.0f}",
            f"  chunks: {metrics.chunk_count}",
            f"  errors: {metrics.errors}",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    source_path: Path,
    audio_seconds: float,
    chunks: list,
    variant_results: dict[str, dict],
    output_root: Path,
    reference_text: str | None = None,
) -> Path:
    stamp = now_stamp()
    output_dir = output_root / safe_stem(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{safe_stem(source_path)}_{stamp}.txt"
    partial_path = final_path.with_suffix(".partial.txt")

    enabled = ", ".join(variant_results.keys())
    lines = [
        "Granite Speech ONNX Transcription + Benchmark",
        f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} local",
        f"Source: {source_path}",
        f"Duration: {format_timestamp(audio_seconds)}",
        "Audio normalized to: 16000 Hz mono WAV",
        f"Chunking: {len(chunks)} chunks, no overlap",
        f"Selected variants: {enabled}",
        "",
    ]

    transcripts: dict[str, str] = {}
    metrics_rows: list[VariantMetrics] = []
    for variant, result in variant_results.items():
        metrics: VariantMetrics = result["metrics"]
        metrics.transcript_path = str(final_path)
        metrics_rows.append(metrics)
        chunk_texts = result.get("chunks", [])
        transcripts[variant] = "\n".join(chunk["text"] for chunk in chunk_texts)
        lines.append(render_variant_section(variant, chunk_texts, metrics, result.get("error")))

    lines.extend(["=" * 80, "COMPARISON SUMMARY", "=" * 80, ""])
    lines.append("Variant      Total Wall  Inference  Speed x-real-time  Peak RAM  Errors")
    for metrics in metrics_rows:
        lines.append(
            f"{metrics.variant:<12} {metrics.total_wall_seconds:>9.3f}s "
            f"{metrics.inference_seconds:>9.3f}s {metrics.audio_seconds_per_wall_second:>8.3f}x "
            f"{metrics.peak_process_memory_mb:>8.0f} MB {metrics.errors:>3}"
        )
    lines.append("")

    comparison: dict[str, dict] = {}
    pairs = [("ar_int8", "ar_fp16w"), ("nar_int8", "nar_fp16w"), ("ar_fp16w", "nar_fp16w")]
    for left, right in pairs:
        if left in transcripts and right in transcripts and transcripts[left] and transcripts[right]:
            metrics = compare_texts(transcripts[left], transcripts[right], reference_text)
            comparison[f"{left}_vs_{right}"] = metrics
            lines.append(f"{left} vs {right}:")
            lines.append(f"  normalized_word_edit_distance: {metrics['normalized_word_edit_distance']:.4f}")
            lines.append(f"  normalized_character_edit_distance: {metrics['normalized_character_edit_distance']:.4f}")
    lines.extend(
        [
            "",
            "Notes:",
            "- Difference metrics compare model outputs to each other; they are not accuracy scores unless a ground-truth transcript is supplied.",
            "- Timestamps are chunk timestamps, not word-level timestamps.",
            "",
        ]
    )

    partial_path.write_text("\n".join(lines), encoding="utf-8")
    partial_path.replace(final_path)

    comparison_path = output_dir / f"{safe_stem(source_path)}_{stamp}_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    write_csv_metrics(output_root / "benchmark_results.csv", metrics_rows)
    return final_path


def write_csv_metrics(path: Path, rows: list[VariantMetrics]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "precision",
                "provider",
                "audio_seconds",
                "chunk_count",
                "model_load_seconds",
                "inference_seconds",
                "total_wall_seconds",
                "tokens_generated",
                "tokens_per_second",
                "real_time_factor",
                "audio_seconds_per_wall_second",
                "peak_process_memory_mb",
                "errors",
                "transcript_path",
            ],
        )
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant": row.variant,
                    "precision": row.precision,
                    "provider": row.provider,
                    "audio_seconds": row.audio_seconds,
                    "chunk_count": row.chunk_count,
                    "model_load_seconds": row.model_load_seconds,
                    "inference_seconds": row.inference_seconds,
                    "total_wall_seconds": row.total_wall_seconds,
                    "tokens_generated": row.tokens_generated,
                    "tokens_per_second": row.tokens_per_second,
                    "real_time_factor": row.real_time_factor,
                    "audio_seconds_per_wall_second": row.audio_seconds_per_wall_second,
                    "peak_process_memory_mb": row.peak_process_memory_mb,
                    "errors": row.errors,
                    "transcript_path": row.transcript_path,
                }
            )
