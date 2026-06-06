from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

from .adapters import BUILTIN_ADAPTERS
from .adapters.base import ModelCandidate
from .config import load_config
from .model_scanner import scan_models
from .model_selector import choose_candidates
from .utils import expand_inputs, sanitize_windows_drag_drop_path, wait_for_stable_file


def print_scan_summary(runnable: list[ModelCandidate], unsupported: list[ModelCandidate]) -> None:
    print("Runnable ASR candidates:")
    if not runnable:
        print("  None")
    for index, candidate in enumerate(runnable, 1):
        print(f"  [{index}] {candidate.display_name} | {candidate.precision} | {candidate.path}")
    print()
    print("Unsupported or incomplete candidates:")
    if not unsupported:
        print("  None")
    for index, candidate in enumerate(unsupported, 1):
        reason = "; ".join(candidate.warnings + ([f"Missing: {', '.join(candidate.missing_files)}"] if candidate.missing_files else []))
        print(f"  [U{index}] {candidate.display_name} | {candidate.container_format} | {reason or 'unsupported'}")


def setup_logging(logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / f"run_{stamp}.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def folder_config(config: dict, key: str, legacy_key: str | None = None) -> Path:
    if "folders" in config and key in config["folders"]:
        return Path(config["folders"][key])
    if legacy_key:
        return Path(config["advanced"][legacy_key])
    return Path(key)


def adapter_for(candidate: ModelCandidate):
    for adapter in BUILTIN_ADAPTERS:
        if adapter.name == candidate.adapter_name:
            return adapter.__class__()
    raise ValueError(f"No adapter registered for {candidate.adapter_name}")


def runtime_config_for_candidate(config: dict) -> dict:
    runtime = dict(config.get("runtime", {}))
    runtime.update(config.get("transcription", {}))
    return runtime


def result_to_report_shape(candidate: ModelCandidate, result, audio_seconds: float) -> dict:
    from .benchmark import VariantMetrics, process_memory_mb

    metrics_dict = dict(result.metrics)
    metrics = VariantMetrics(
        variant=candidate.candidate_id,
        precision=candidate.precision,
        provider=str(metrics_dict.get("provider", "")),
        audio_seconds=audio_seconds,
        chunk_count=int(metrics_dict.get("chunk_count", len(result.transcript_chunks))),
        model_load_seconds=float(metrics_dict.get("model_load_seconds", 0.0)),
        preprocessing_seconds=float(metrics_dict.get("preprocessing_seconds", 0.0)),
        inference_seconds=float(metrics_dict.get("inference_seconds", 0.0)),
        total_wall_seconds=float(metrics_dict.get("total_wall_seconds", 0.0)),
        tokens_generated=int(metrics_dict.get("tokens_generated", 0)),
        peak_process_memory_mb=float(metrics_dict.get("peak_process_memory_mb", process_memory_mb())),
        errors=len(result.errors),
    )
    chunks = [
        {
            "index": index,
            "start_seconds": chunk.start_seconds,
            "end_seconds": chunk.end_seconds,
            "text": chunk.text,
        }
        for index, chunk in enumerate(result.transcript_chunks)
    ]
    return {"chunks": chunks, "metrics": metrics, "error": "\n".join(result.errors) if result.errors else None}


def process_file_with_candidates(source: Path, candidates: list[ModelCandidate], config: dict) -> Path | None:
    from .benchmark import VariantMetrics, process_memory_mb, timer
    from .media import audio_duration_seconds, prepare_audio
    from .output_writer import write_report

    source = source.resolve()
    temp_dir = folder_config(config, "temp", "temp_folder")
    output_dir = folder_config(config, "output")
    logging.info("Processing %s", source)
    try:
        wait_for_stable_file(source, float(config["input"]["file_stability_wait_seconds"]))
        with timer() as preprocess_elapsed:
            wav_path, samples, chunks = prepare_audio(source, temp_dir, config)
            audio_seconds = audio_duration_seconds(samples)
        chunk_metadata = [
            {
                "chunk_id": f"{chunk.index + 1:04d}",
                "start_seconds": chunk.start_seconds,
                "end_seconds": chunk.end_seconds,
            }
            for chunk in chunks
        ]
        report_results: dict[str, dict] = {}
        for candidate in candidates:
            logging.info("Running %s", candidate.candidate_id)
            adapter = adapter_for(candidate)
            try:
                adapter.load(candidate, runtime_config_for_candidate(config))
                result = adapter.transcribe_chunks(chunks, chunk_metadata)
                shaped = result_to_report_shape(candidate, result, audio_seconds)
                shaped["metrics"].preprocessing_seconds = preprocess_elapsed()
                report_results[candidate.candidate_id] = shaped
            except Exception:
                logging.error("%s failed", candidate.candidate_id)
                logging.debug(traceback.format_exc())
                report_results[candidate.candidate_id] = {
                    "chunks": [],
                    "error": traceback.format_exc(),
                    "metrics": VariantMetrics(
                        variant=candidate.candidate_id,
                        precision=candidate.precision,
                        provider=str(config["runtime"].get("provider", "auto")),
                        audio_seconds=audio_seconds,
                        chunk_count=len(chunks),
                        preprocessing_seconds=preprocess_elapsed(),
                        errors=1,
                        peak_process_memory_mb=process_memory_mb(),
                    ),
                }
            finally:
                adapter.unload()
        output_path = write_report(source, audio_seconds, chunks, report_results, output_dir)
        if wav_path.exists() and not config["advanced"].get("keep_temp_wavs", False):
            wav_path.unlink()
        print(f"Wrote {output_path}")
        return output_path
    except Exception:
        logging.error("Failed processing %s", source)
        logging.debug(traceback.format_exc())
        print(traceback.format_exc())
        return None


def collect_input_files(args: argparse.Namespace, config: dict) -> list[Path]:
    raw_paths = [sanitize_windows_drag_drop_path(path) for path in args.paths]
    if not raw_paths:
        input_dir = folder_config(config, "input")
        raw_paths = [input_dir]
    extensions = {ext.lower() for ext in config["input"]["extensions"]}
    return expand_inputs(raw_paths, extensions, bool(config["input"]["recursive_folders"]))


def interactive_prompt_for_files(config: dict) -> list[Path]:
    print()
    print("Drop/paste audio or video file/folder paths. Press Enter with a blank line to use Input/.")
    print("Type q to quit.")
    raw = input("Input> ").strip()
    if raw.lower() in {"q", "quit", "exit"}:
        return []
    if raw:
        paths = [sanitize_windows_drag_drop_path(raw)]
    else:
        paths = [folder_config(config, "input")]
    extensions = {ext.lower() for ext in config["input"]["extensions"]}
    return expand_inputs(paths, extensions, bool(config["input"]["recursive_folders"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    models_root = folder_config(config, "models")
    runnable, unsupported = scan_models(models_root)
    if args.scan_only:
        print_scan_summary(runnable, unsupported)
        return
    setup_logging(folder_config(config, "logs", "logs_folder"))
    selected = choose_candidates(runnable, unsupported) if args.interactive and not args.scan_only else runnable
    if not selected:
        print("No runnable ASR models selected.")
        return
    while True:
        files = collect_input_files(args, config) if args.paths else interactive_prompt_for_files(config)
        if not files:
            return
        for file_path in files:
            process_file_with_candidates(file_path, selected, config)
        if not args.interactive:
            return
        args.paths = []


if __name__ == "__main__":
    main()
