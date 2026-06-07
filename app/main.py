from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from .adapters import BUILTIN_ADAPTERS
from .adapters.base import ModelCandidate, ModelRunResult
from .config import load_config
from .console_style import key, prompt_label
from .hf_model_downloader import download_hf_model_interactive
from .interactive_menu import MenuAction, choose_one
from .model_scanner import scan_models
from .model_selector import choose_candidates
from .model_status import candidate_reason, model_status_label
from .utils import expand_inputs, parse_windows_path_list, sanitize_windows_drag_drop_path, wait_for_stable_file


def print_scan_summary(runnable: list[ModelCandidate], unsupported: list[ModelCandidate]) -> None:
    buckets = [
        ("Runnable ASR candidates", [candidate for candidate in runnable if candidate.category == "asr"]),
        ("Needs dependency install", [candidate for candidate in unsupported if model_status_label(candidate) == "Needs dependency install"]),
        ("ASR probe required", [candidate for candidate in unsupported if model_status_label(candidate) == "ASR probe required"]),
        ("Reference/correction LLM candidates", [candidate for candidate in runnable + unsupported if candidate.category == "reference_llm"]),
        ("Recognized incomplete", [candidate for candidate in unsupported if model_status_label(candidate) == "Recognized incomplete"]),
        ("Unsafe blocked", [candidate for candidate in unsupported if model_status_label(candidate) == "Unsafe blocked"]),
        ("Unsupported or inspection-only", [
            candidate
            for candidate in unsupported
            if candidate.category != "reference_llm"
            and model_status_label(candidate) not in {"Needs dependency install", "ASR probe required", "Recognized incomplete", "Unsafe blocked"}
        ]),
    ]
    for title, items in buckets:
        print(f"{title}:")
        if not items:
            print("  None")
        for index, candidate in enumerate(items, 1):
            if candidate.runnable:
                print(f"  [{index}] {candidate.display_name} | {candidate.precision} | {candidate.path}")
            else:
                print(f"  [{index}] {candidate.display_name} | {candidate.container_format} | {candidate_reason(candidate)}")
        print()


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


def latest_log_path(config: dict) -> Path | None:
    try:
        logs_dir = folder_config(config, "logs", "logs_folder")
    except Exception:
        logs_dir = Path("Logs")
    candidates = sorted(logs_dir.glob("run_*.log"), key=lambda path: path.stat().st_mtime, reverse=True) if logs_dir.exists() else []
    return candidates[0] if candidates else None


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
    from .dependency_manager import acceleration_install_decision, dependency_status, install_group_for_config, missing_modules_for_config, recovery_command_for_config
    from .install_plan import build_install_plan, format_install_plan

    project_root = Path(__file__).resolve().parent.parent
    status = dependency_status()
    candidate_groups: dict[str, list[str]] = {}
    all_candidates = [*candidates, *([reference_llm] if reference_llm else [])]
    groups: list[str] = []
    for candidate in all_candidates:
        adapter = adapter_for(candidate)
        candidate_groups[candidate.candidate_id] = adapter.required_dependency_groups(candidate)
        for group in adapter.required_dependency_groups(candidate):
            if group not in groups:
                groups.append(group)
    missing = {group: missing_modules_for_config(group, config) for group in groups if missing_modules_for_config(group, config)}
    if not missing:
        return candidates, reference_llm
    print()
    print("Some selected models need additional runtime packages:")
    group_candidates: dict[str, list[str]] = {group: [] for group in groups}
    for candidate in all_candidates:
        for group in candidate_groups.get(candidate.candidate_id, []):
            group_candidates.setdefault(group, []).append(candidate.display_name)
    for group, modules in missing.items():
        detail = status.get(group, {})
        description = detail.get("description", "optional runtime support")
        acceleration_decision = acceleration_install_decision(config, group)
        print(f"  {group}: {description}")
        print(f"    missing: {', '.join(modules)}")
        if acceleration_decision["use_accelerator"]:
            print(f"    {acceleration_decision['accelerator'].upper()} install: {acceleration_decision['reason']}")
            print(f"    repair: {recovery_command_for_config(group, config)}")
        else:
            print(f"    repair: {recovery_command_for_config(group, config)}")
            if "accelerator" in acceleration_decision["reason"].lower() or "gpu" in acceleration_decision["reason"].lower() or "nvidia" in acceleration_decision["reason"].lower():
                print(f"    accelerator note: {acceleration_decision['reason']}")
        print(format_install_plan(build_install_plan(group, project_root, config, group_candidates.get(group, []), _dependency_install_log_path(config, group))))
    if not config.get("dependency_install", {}).get("auto_install_missing_runtime_dependencies", True):
        print("Automatic dependency repair is disabled in config.json.")
        failed_groups = set(missing)
        return _drop_candidates_for_failed_dependency_groups(candidates, reference_llm, candidate_groups, failed_groups)
    failed_groups: set[str] = set()
    for group in missing:
        acceleration_decision = acceleration_install_decision(config, group)
        log_path = _dependency_install_log_path(config, group)
        plan = build_install_plan(group, project_root, config, group_candidates.get(group, []), log_path)
        decision = _dependency_install_confirmation(group, recovery_command_for_config(group, config))
        if decision == "quit":
            print(f"Quit requested while installing {group}. Skipping all remaining models that need missing dependency groups.")
            failed_groups.update(missing)
            break
        if decision != "install":
            print(f"Skipped dependency install for {group}. Only models requiring this dependency group are skipped; other selected models continue.")
            failed_groups.add(group)
            continue
        accelerator = acceleration_decision.get("accelerator") if acceleration_decision["use_accelerator"] else ""
        install_label = f"{group} with {str(accelerator).upper()} packages" if accelerator else group
        print(f"Installing {install_label}...")
        try:
            install_decision = install_group_for_config(group, project_root, config, log_path=Path(log_path))
        except Exception as exc:
            print(f"Install failed for {group}: {exc}")
            print(f"Install log: {log_path}")
            print(f"Manual repair command: {recovery_command_for_config(group, config)}")
            failed_groups.add(group)
            continue
        install_decision = install_decision or {}
        if install_decision.get("accelerator_fallback_reason"):
            print(f"Accelerator fallback: {install_decision['accelerator_fallback_reason']}")
        still_missing = missing_modules_for_config(group, config)
        if still_missing:
            print(f"Install finished but {group} is still missing: {', '.join(still_missing)}")
            print(f"Manual repair command: {recovery_command_for_config(group, config)}")
            failed_groups.add(group)
    return _drop_candidates_for_failed_dependency_groups(candidates, reference_llm, candidate_groups, failed_groups)


def _dependency_install_log_path(config: dict, group: str) -> str:
    if "folders" in config and "logs" in config["folders"]:
        logs_dir = Path(config["folders"]["logs"])
    elif "advanced" in config and "logs_folder" in config["advanced"]:
        logs_dir = Path(config["advanced"]["logs_folder"])
    else:
        logs_dir = Path("Logs")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return str(logs_dir / f"dependency_install_{group}_{stamp}.log")


def _dependency_install_confirmation(group: str, repair_command: str | None = None) -> str:
    if not sys.stdin.isatty():
        print(f"Skipping dependency install for {group}: noninteractive input cannot confirm optional installs.")
        return "skip_group"
    menu_result = choose_one(
        f"Install missing runtime package group: {group}",
        [
            "Install now",
            "Skip affected models",
            "Show repair command",
            "Quit batch",
        ],
        actions=[MenuAction("I", "install"), MenuAction("S", "skip affected models"), MenuAction("R", "show repair command"), MenuAction("Q", "quit batch")],
    )
    if menu_result == 0 or menu_result == "i":
        return "install"
    if menu_result == 1 or menu_result == "s":
        return "skip_group"
    if menu_result == 3 or menu_result == "q":
        return "quit"
    if menu_result == 2 or menu_result == "r":
        print(f"Manual repair command: {repair_command or 'Run setup.bat --doctor for a repair command.'}")
    try:
        answer = input(
            f"{key('I')} install, {key('S')} skip affected models, {key('R')} show repair command, {key('Q')} quit batch: "
        ).strip().lower()
    except EOFError:
        print(f"Skipping dependency install for {group}: input closed before confirmation.")
        return "skip_group"
    if answer in {"", "i", "install", "y", "yes"}:
        return "install"
    if answer in {"s", "skip", "skip affected", "skip affected models", "n", "no"}:
        return "skip_group"
    if answer in {"r", "repair", "show repair"}:
        print(f"Manual repair command: {repair_command or 'Run setup.bat --doctor for a repair command.'}")
        return _dependency_install_confirmation(group, repair_command)
    if answer in {"q", "quit", "exit"}:
        return "quit"
    print(f"Unrecognized choice for {group}; skipping affected models so the rest of the batch can continue.")
    return "skip_group"


def warn_runtime_dependency_fallbacks(config: dict) -> None:
    runtime = config.get("runtime", {})
    provider = str(runtime.get("provider", "auto")).lower()
    prefer_gpu = bool(runtime.get("prefer_gpu", False))
    if provider != "cuda" and not prefer_gpu:
        return
    from .dependency_manager import cuda_diagnostics

    diagnostics = cuda_diagnostics()
    messages = diagnostics.get("messages", [])
    if not messages:
        return
    print()
    print("CUDA was requested or preferred, but this install may fall back to CPU:")
    for message in messages:
        print(f"  - {message}")
    print("Run setup.bat --doctor for full dependency status and repair commands.")


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


def classify_file_failure(exc: Exception) -> tuple[str, list[str], list[str]]:
    message = str(exc).lower()
    if "no audio stream" in message:
        return (
            "media_probe",
            [
                "The video has no audio track.",
                "The audio stream is unsupported, missing, or corrupt.",
            ],
            [
                "Try a file with an audio stream.",
                "If this is expected to contain audio, check it in a media player or extract audio with another tool.",
            ],
        )
    if "ffmpeg" in message or "ffprobe" in message or "decode" in message or "conversion failed" in message:
        return (
            "ffmpeg_convert",
            [
                "FFmpeg could not inspect or decode this media file.",
                "The media file may be corrupt, encrypted, unsupported, or incomplete.",
            ],
            [
                "Try converting the file to WAV, MP3, or MP4 with a known-good tool.",
                "Run setup.bat --doctor to verify media dependencies.",
            ],
        )
    if isinstance(exc, (FileNotFoundError, PermissionError, OSError)):
        return (
            "path_or_file_access",
            [
                "The file path is unavailable, locked, or no longer readable.",
                "The output, temp, or input folder may not be writable.",
            ],
            [
                "Move the file to a local folder such as Input and try again.",
                "Close programs that may be locking the file and confirm folder permissions.",
            ],
        )
    return (
        "pre_model_processing",
        [
            "The file path is unavailable, locked, or no longer readable.",
            "The media file has no supported audio stream, is corrupt, or FFmpeg could not decode it.",
            "A preprocessing dependency failed before model benchmarking could start.",
        ],
        [
            "Try a known-good WAV, MP3, or MP4 with an audio track.",
            "If this is a video, confirm it has an audio stream.",
            "Check the linked log for the exact preprocessing error.",
        ],
    )


def process_file_with_candidates(
    source: Path,
    candidates: list[ModelCandidate],
    config: dict,
    unsupported_models: list[ModelCandidate] | None = None,
    reference_llm: ModelCandidate | None = None,
) -> Path | None:
    from .benchmark import peak_vram_mb, process_memory_mb, reset_peak_vram, timer
    from .media import audio_duration_seconds, prepare_audio
    from .results_writer import build_failed_file_results, build_results, write_all_reports

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
                reset_peak_vram()
                adapter.load(candidate, runtime_config_for_candidate(config))
                model_load_seconds = time.perf_counter() - load_started
                result = adapter.transcribe_chunks(chunks, chunk_metadata)
                result.metrics["model_load_seconds"] = model_load_seconds
                result.metrics["total_wall_seconds"] = model_load_seconds + float(result.metrics.get("inference_seconds", 0))
                audio_seconds_metric = float(result.metrics.get("audio_seconds", audio_seconds))
                result.metrics["audio_seconds_per_wall_second"] = audio_seconds_metric / max(0.001, float(result.metrics["total_wall_seconds"]))
                result.metrics["peak_vram_mb"] = peak_vram_mb()
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
        print(f"Open HTML comparison: {output_path / 'compare.html'}")
        return output_path
    except Exception as exc:
        logging.error("Failed processing %s", source)
        logging.exception("File failed")
        stage, likely_causes, next_actions = classify_file_failure(exc)
        try:
            failed_results = build_failed_file_results(
                source_path=source,
                stage=stage,
                error_type=exc.__class__.__name__,
                message=str(exc) or exc.__class__.__name__,
                likely_causes=likely_causes,
                next_actions=next_actions,
                log_path=latest_log_path(config),
                selected_models=candidates,
                unsupported_models=unsupported_models,
            )
            output_path = write_all_reports(failed_results, output_dir)
            print(f"File failed before model benchmarking. Wrote failure report to {output_path}")
            print(f"Open failure report: {output_path / 'compare.html'}")
            return output_path
        except Exception:
            logging.exception("Failed writing failed-file report")
            print("File failed and the failure report could not be written. See the latest log in Logs for the full error.")
            return None


def collect_input_files(args: argparse.Namespace, config: dict) -> list[Path]:
    raw_paths = [sanitize_windows_drag_drop_path(path) for path in args.paths]
    if not raw_paths:
        input_dir = folder_config(config, "input")
        raw_paths = [input_dir]
    extensions = {ext.lower() for ext in config["input"]["extensions"]}
    files, skipped = expand_inputs(raw_paths, extensions, bool(config["input"]["recursive_folders"]), include_skipped=True)
    for path in skipped:
        print(f"Skipping unsupported input extension: {path}")
        logging.info("Skipping unsupported input extension: %s", path)
    return files


def queue_state(config: dict):
    from .queue_manager import QueueState

    return QueueState(folder_config(config, "logs") / "state.json")


def interactive_prompt_for_files(config: dict) -> list[Path]:
    print()
    print("Drop/paste audio or video file/folder paths. Press Enter with a blank line to use Input/.")
    print(f"Type {key('q')} to quit.")
    raw = input(prompt_label("Input> ")).strip()
    if raw.lower() in {"q", "quit", "exit"}:
        return []
    if raw:
        paths = parse_windows_path_list(raw)
    else:
        paths = [folder_config(config, "input")]
    extensions = {ext.lower() for ext in config["input"]["extensions"]}
    files, skipped = expand_inputs(paths, extensions, bool(config["input"]["recursive_folders"]), include_skipped=True)
    for path in skipped:
        print(f"Skipping unsupported input extension: {path}")
        logging.info("Skipping unsupported input extension: %s", path)
    return files


def print_first_run_guidance(config: dict, runnable: list[ModelCandidate], unsupported: list[ModelCandidate], models_root: Path) -> None:
    print()
    print("Easy ASR Bench")
    print("Models and media stay local. Network is used only for setup, optional packages, or a model download you choose.")
    if runnable:
        print()
        print("Runnable ASR model found.")
        print(f"Put audio/video files in {folder_config(config, 'input')} or paste paths when prompted.")
        return
    incomplete = [candidate for candidate in unsupported if model_status_label(candidate) == "Recognized incomplete"]
    reference_llms = [candidate for candidate in unsupported if candidate.category == "reference_llm"]
    print()
    print("No runnable ASR model is installed yet.")
    print("Detected files:")
    print(f"  - {len(incomplete)} incomplete model folder(s)")
    print(f"  - {len(reference_llms)} reference/correction LLM(s)")
    print("  - 0 runnable ASR models")
    print()
    print("Next steps:")
    print(f"  [{key('P')}] Paste a Hugging Face model/repo link from the model downloader option")
    print(f"  [{key('M')}] Open {models_root} and add a complete ASR model folder, then run again")
    print(f"  [{key('I')}] Put audio/video files in {folder_config(config, 'input')}")
    print(f"  [{key('Q')}] Quit")
    print()
    print("Unsupported or incomplete files are shown separately and will not be used as ASR models.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--first-run", action="store_true")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--open-models", action="store_true")
    parser.add_argument("--open-input", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    try:
        _main(args)
    except KeyboardInterrupt:
        print()
        print("Stopped by user. Queue state is saved in Logs/state.json.")


def _main(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    if args.doctor:
        from .doctor import run_doctor

        raise SystemExit(run_doctor(Path(args.config), strict=False))
    models_root = folder_config(config, "models")
    if args.open_models:
        open_folder(models_root)
        return
    if args.open_input:
        open_folder(folder_config(config, "input"))
        return
    if args.download_model:
        destination = download_hf_model_interactive(models_root)
        if destination is not None:
            print(f"Model package saved. Run Easy ASR Bench again to rescan: {destination}")
        return
    if args.first_run:
        from .first_run import run_first_run_wizard

        if not run_first_run_wizard(config):
            return
        args.interactive = True
    runnable, unsupported = scan_models(models_root)
    if args.interactive:
        print_first_run_guidance(config, runnable, unsupported, models_root)
    if args.scan_only:
        print_scan_summary(runnable, unsupported)
        return
    setup_logging(folder_config(config, "logs", "logs_folder"))
    warn_runtime_dependency_fallbacks(config)
    reference_llm = None
    if args.interactive and not args.scan_only:
        selected, reference_llm = choose_candidates(runnable, unsupported, config, Path(args.config), models_root)
    else:
        selected = [candidate for candidate in runnable if candidate.category == "asr"]
    if not selected:
        print("No runnable ASR models selected.")
        return
    selected, reference_llm = ensure_dependencies(selected, config, reference_llm)
    if not selected:
        print("Cannot run selected models until their dependency groups are installed.")
        return
    batch_rows: list[dict] = []
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
                    status = output_status(output)
                    state.mark(file_path.resolve(), status, str(output or ""))
                    batch_rows.append({"source_path": str(file_path.resolve()), "status": status, "output_path": str(output or "")})
                if args.once and len(batch_rows) > 1:
                    write_batch_summary(config, batch_rows)
                if args.once:
                    return
                time.sleep(2)
        files = collect_input_files(args, config) if args.paths else interactive_prompt_for_files(config)
        if not files:
            return
        state = queue_state(config)
        for file_path in files:
            from .queue_manager import QueueItem
            from .utils import file_key

            state.upsert(QueueItem(str(file_path.resolve()), "", file_key(file_path)))
            output = process_file_with_candidates(file_path, selected, config, unsupported, reference_llm)
            status = output_status(output)
            state.mark(file_path.resolve(), status, str(output or ""))
            batch_rows.append({"source_path": str(file_path.resolve()), "status": status, "output_path": str(output or "")})
        if len(files) > 1:
            write_batch_summary(config, batch_rows[-len(files) :])
        if not args.interactive:
            return
        args.paths = []


def write_batch_summary(config: dict, rows: list[dict]) -> None:
    from .batch_report import write_batch_report

    output_dir = write_batch_report(folder_config(config, "output"), rows)
    print(f"Wrote batch overview to {output_dir}")
    print(f"Open batch dashboard: {output_dir / 'index.html'}")


def open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    print(path)


def output_status(output_path: Path | None) -> str:
    if output_path is None:
        return "failed"
    results_path = output_path / "results.json"
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "done"
    if results.get("errors") and not results.get("runs"):
        return "failed"
    return "done"


if __name__ == "__main__":
    main()
