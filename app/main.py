from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

from .adapters import BUILTIN_ADAPTERS
from .adapters.base import ModelCandidate, ModelRunResult
from .config import load_config
from .model_scanner import scan_models
from .model_selector import choose_candidates
from .utils import expand_inputs, parse_windows_path_list, sanitize_windows_drag_drop_path, wait_for_stable_file


def print_scan_summary(runnable: list[ModelCandidate], unsupported: list[ModelCandidate]) -> None:
    print("Runnable ASR candidates:")
    asr = [candidate for candidate in runnable if candidate.category == "asr"]
    refs = [candidate for candidate in runnable + unsupported if candidate.category == "reference_llm"]
    if not asr:
        print("  None")
    for index, candidate in enumerate(asr, 1):
        print(f"  [{index}] {candidate.display_name} | {candidate.precision} | {candidate.path}")
    print()
    print("Reference/correction LLM candidates:")
    if not refs:
        print("  None")
    for index, candidate in enumerate(refs, 1):
        print(f"  [L{index}] {candidate.display_name} | {candidate.precision} | {candidate.path}")
    print()
    print("Unsupported or incomplete candidates:")
    unsupported_only = [candidate for candidate in unsupported if candidate.category != "reference_llm"]
    if not unsupported_only:
        print("  None")
    for index, candidate in enumerate(unsupported_only, 1):
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


def ensure_dependencies(candidates: list[ModelCandidate], config: dict, reference_llm: ModelCandidate | None = None) -> tuple[list[ModelCandidate], ModelCandidate | None]:
    from .dependency_manager import install_group, missing_modules

    project_root = Path(__file__).resolve().parent.parent
    candidate_groups: dict[str, list[str]] = {}
    all_candidates = [*candidates, *([reference_llm] if reference_llm else [])]
    groups: list[str] = []
    for candidate in all_candidates:
        adapter = adapter_for(candidate)
        candidate_groups[candidate.candidate_id] = adapter.required_dependency_groups(candidate)
        for group in adapter.required_dependency_groups(candidate):
            if group not in groups:
                groups.append(group)
    missing = {group: missing_modules(group) for group in groups if missing_modules(group)}
    if not missing:
        return candidates, reference_llm
    print()
    print("Some selected models need additional runtime packages:")
    for group, modules in missing.items():
        print(f"  {group}: missing {', '.join(modules)}")
    if not config.get("dependency_install", {}).get("auto_install_missing_runtime_dependencies", True):
        print("Automatic dependency repair is disabled in config.json.")
        failed_groups = set(missing)
        return _drop_candidates_for_failed_dependency_groups(candidates, reference_llm, candidate_groups, failed_groups)
    answer = input("Install missing dependency groups now? [Y/n] ").strip().lower()
    if answer in {"n", "no"}:
        failed_groups = set(missing)
        return _drop_candidates_for_failed_dependency_groups(candidates, reference_llm, candidate_groups, failed_groups)
    failed_groups: set[str] = set()
    for group in missing:
        print(f"Installing {group}...")
        try:
            install_group(group, project_root)
        except Exception as exc:
            print(f"Install failed for {group}: {exc}")
            failed_groups.add(group)
            continue
        still_missing = missing_modules(group)
        if still_missing:
            print(f"Install finished but {group} is still missing: {', '.join(still_missing)}")
            failed_groups.add(group)
    return _drop_candidates_for_failed_dependency_groups(candidates, reference_llm, candidate_groups, failed_groups)


def _drop_candidates_for_failed_dependency_groups(
    candidates: list[ModelCandidate],
    reference_llm: ModelCandidate | None,
    candidate_groups: dict[str, list[str]],
    failed_groups: set[str],
) -> tuple[list[ModelCandidate], ModelCandidate | None]:
    if not failed_groups:
        return candidates, reference_llm
    kept: list[ModelCandidate] = []
    for candidate in candidates:
        if failed_groups & set(candidate_groups.get(candidate.candidate_id, [])):
            print(f"Skipping {candidate.display_name}: dependency install failed for {', '.join(sorted(failed_groups & set(candidate_groups.get(candidate.candidate_id, []))))}.")
        else:
            kept.append(candidate)
    if reference_llm and failed_groups & set(candidate_groups.get(reference_llm.candidate_id, [])):
        print(f"Skipping {reference_llm.display_name}: dependency install failed for {', '.join(sorted(failed_groups & set(candidate_groups.get(reference_llm.candidate_id, []))))}.")
        reference_llm = None
    return kept, reference_llm


def runtime_config_for_candidate(config: dict) -> dict:
    runtime = dict(config.get("runtime", {}))
    runtime.update(config.get("transcription", {}))
    runtime["security"] = config.get("security", {})
    runtime["whisper"] = config.get("whisper", {})
    return runtime


def process_file_with_candidates(
    source: Path,
    candidates: list[ModelCandidate],
    config: dict,
    unsupported_models: list[ModelCandidate] | None = None,
    reference_llm: ModelCandidate | None = None,
) -> Path | None:
    from .benchmark import process_memory_mb, timer
    from .media import audio_duration_seconds, prepare_audio
    from .results_writer import build_results, write_all_reports

    source = source.resolve()
    temp_dir = folder_config(config, "temp", "temp_folder")
    output_dir = folder_config(config, "output")
    logging.info("Processing %s", source)
    wav_path = None
    try:
        wait_for_stable_file(source, float(config["input"]["file_stability_wait_seconds"]))
        with timer() as preprocess_elapsed:
            wav_path, samples, chunks = prepare_audio(source, temp_dir, config)
            audio_seconds = audio_duration_seconds(samples)
            media_seconds = preprocess_elapsed()
        chunk_metadata = [
            {
                "chunk_id": f"{chunk.index + 1:04d}",
                "start_seconds": chunk.start_seconds,
                "end_seconds": chunk.end_seconds,
            }
            for chunk in chunks
        ]
        run_results: list[ModelRunResult] = []
        for candidate in candidates:
            logging.info("Running %s", candidate.candidate_id)
            adapter = adapter_for(candidate)
            try:
                load_started = time.perf_counter()
                adapter.load(candidate, runtime_config_for_candidate(config))
                model_load_seconds = time.perf_counter() - load_started
                result = adapter.transcribe_chunks(chunks, chunk_metadata)
                result.metrics["model_load_seconds"] = model_load_seconds
                result.metrics["total_wall_seconds"] = model_load_seconds + float(result.metrics.get("inference_seconds", 0))
                audio_seconds_metric = float(result.metrics.get("audio_seconds", audio_seconds))
                result.metrics["audio_seconds_per_wall_second"] = audio_seconds_metric / max(0.001, float(result.metrics["total_wall_seconds"]))
            except Exception:
                logging.error("%s failed", candidate.candidate_id)
                logging.exception("Model failed")
                result = ModelRunResult(
                    candidate=candidate,
                    transcript_chunks=[],
                    metrics={
                        "provider": str(config["runtime"].get("provider", "auto")),
                        "audio_seconds": audio_seconds,
                        "chunk_count": len(chunks),
                        "media_normalization_seconds": media_seconds,
                        "peak_process_memory_mb": process_memory_mb(),
                    },
                    errors=[traceback.format_exc()],
                )
            finally:
                adapter.unload()
            run_results.append(result)
        unsupported_for_report = list(unsupported_models or [])
        if reference_llm:
            unsupported_for_report.append(reference_llm)
        results = build_results(source, audio_seconds, chunks, run_results, unsupported_for_report, media_seconds)
        if reference_llm:
            results["reference_llm"] = {
                "candidate_id": reference_llm.candidate_id,
                "display_name": reference_llm.display_name,
                "path": str(reference_llm.path),
                "status": "selected_for_llm_corrected_reference",
                "note": "GGUF text LLMs are used for reference/correction, not direct ASR.",
            }
            try:
                ref_adapter = adapter_for(reference_llm)
                if hasattr(ref_adapter, "generate_reference"):
                    results["local_llm_reference_attempt"] = ref_adapter.generate_reference(reference_llm, runtime_config_for_candidate(config), results)
            except Exception as exc:
                results["local_llm_reference_attempt"] = {
                    "candidate_id": reference_llm.candidate_id,
                    "display_name": reference_llm.display_name,
                    "status": "failed",
                    "error": str(exc),
                }
        output_path = write_all_reports(results, output_dir)
        if wav_path and wav_path.exists() and not config["advanced"].get("keep_temp_wavs", False):
            wav_path.unlink()
        print(f"Wrote reports to {output_path}")
        return output_path
    except Exception:
        logging.error("Failed processing %s", source)
        logging.exception("File failed")
        print("File failed. See the latest log in Logs for the full error.")
        return None


def collect_input_files(args: argparse.Namespace, config: dict) -> list[Path]:
    raw_paths = [sanitize_windows_drag_drop_path(path) for path in args.paths]
    if not raw_paths:
        input_dir = folder_config(config, "input")
        raw_paths = [input_dir]
    extensions = {ext.lower() for ext in config["input"]["extensions"]}
    return expand_inputs(raw_paths, extensions, bool(config["input"]["recursive_folders"]))


def queue_state(config: dict):
    from .queue_manager import QueueState

    return QueueState(folder_config(config, "logs") / "state.json")


def interactive_prompt_for_files(config: dict) -> list[Path]:
    print()
    print("Drop/paste audio or video file/folder paths. Press Enter with a blank line to use Input/.")
    print("Type q to quit.")
    raw = input("Input> ").strip()
    if raw.lower() in {"q", "quit", "exit"}:
        return []
    if raw:
        paths = parse_windows_path_list(raw)
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
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.doctor:
        from .doctor import run_doctor

        raise SystemExit(run_doctor(Path(args.config), strict=False))
    models_root = folder_config(config, "models")
    runnable, unsupported = scan_models(models_root)
    if args.scan_only:
        print_scan_summary(runnable, unsupported)
        return
    setup_logging(folder_config(config, "logs", "logs_folder"))
    reference_llm = None
    if args.interactive and not args.scan_only:
        selected, reference_llm = choose_candidates(runnable, unsupported, config, Path(args.config))
    else:
        selected = [candidate for candidate in runnable if candidate.category == "asr"]
    if not selected:
        print("No runnable ASR models selected.")
        return
    selected, reference_llm = ensure_dependencies(selected, config, reference_llm)
    if not selected:
        print("Cannot run selected models until their dependency groups are installed.")
        return
    while True:
        if args.watch:
            from .queue_manager import discover_queue

            print("Watching Input for supported media. Press Ctrl+C to stop.")
            state = queue_state(config)
            while True:
                files = discover_queue(
                    [folder_config(config, "input")],
                    {ext.lower() for ext in config["input"]["extensions"]},
                    bool(config["input"]["recursive_folders"]),
                    float(config["input"]["file_stability_wait_seconds"]),
                    state,
                    bool(config["input"].get("skip_already_processed_by_hash", True)),
                )
                for file_path in files:
                    output = process_file_with_candidates(file_path, selected, config, unsupported, reference_llm)
                    state.mark(file_path.resolve(), "done" if output else "failed", str(output or ""))
                if args.once:
                    return
                time.sleep(2)
        files = collect_input_files(args, config) if args.paths else interactive_prompt_for_files(config)
        if not files:
            return
        state = queue_state(config)
        for file_path in files:
            from .queue_manager import QueueItem
            from .utils import file_key, sha256_file

            state.upsert(QueueItem(str(file_path.resolve()), sha256_file(file_path), file_key(file_path)))
            output = process_file_with_candidates(file_path, selected, config, unsupported, reference_llm)
            state.mark(file_path.resolve(), "done" if output else "failed", str(output or ""))
        if not args.interactive:
            return
        args.paths = []


if __name__ == "__main__":
    main()
