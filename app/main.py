from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import importlib.metadata
from pathlib import Path

from .adapters import BUILTIN_ADAPTERS
from .adapters.base import ModelCandidate, ModelRunResult
from .config import load_config
from .console_style import key, prompt_label
from .hf_model_downloader import download_hf_model_interactive
from .interactive_menu import MenuAction, choose_one
from .model_scanner import scan_models
from .model_selector import choose_candidates, resolve_last_run_selection
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
    missing: dict[str, list[str]] = {}
    for group in groups:
        group_missing = missing_modules_for_config(group, config)
        if group_missing:
            missing[group] = group_missing
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
    repair_commands = {group: recovery_command_for_config(group, config) for group in missing}
    decision = _dependency_install_batch_confirmation(list(missing), repair_commands)
    if decision == "quit":
        print("Quit requested before dependency installs. No selected models that need missing dependency groups will run.")
        return _drop_candidates_for_failed_dependency_groups(candidates, reference_llm, candidate_groups, set(missing))
    if decision != "install":
        print("Dependency install was not confirmed before processing. Models requiring missing dependency groups will not run.")
        return _drop_candidates_for_failed_dependency_groups(candidates, reference_llm, candidate_groups, set(missing))
    failed_groups: set[str] = set()
    for group in missing:
        acceleration_decision = acceleration_install_decision(config, group)
        log_path = _dependency_install_log_path(config, group)
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
    kept, reference_llm = _drop_candidates_for_failed_dependency_groups(candidates, reference_llm, candidate_groups, failed_groups)
    kept = _repair_native_backend_preflights(kept, config)
    return kept, reference_llm


def _dependency_install_log_path(config: dict, group: str) -> str:
    if "folders" in config and "logs" in config["folders"]:
        logs_dir = Path(config["folders"]["logs"])
    elif "advanced" in config and "logs_folder" in config["advanced"]:
        logs_dir = Path(config["advanced"]["logs_folder"])
    else:
        logs_dir = Path("Logs")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return str(logs_dir / f"dependency_install_{group}_{stamp}.log")


def _dependency_install_batch_confirmation(groups: list[str], repair_commands: dict[str, str] | None = None) -> str:
    group_label = ", ".join(groups)
    if not sys.stdin.isatty():
        print(
            "Skipping dependency installs before processing: noninteractive input cannot confirm optional installs "
            f"for {group_label}."
        )
        return "skip_all"
    menu_result = choose_one(
        "Install missing runtime package groups before processing",
        [
            f"Install all now ({group_label})",
            "Skip affected models",
            "Show repair commands",
            "Quit batch",
        ],
        actions=[MenuAction("I", "install all"), MenuAction("S", "skip affected models"), MenuAction("R", "show repair commands"), MenuAction("Q", "quit batch")],
    )
    if menu_result == 0 or menu_result == "i":
        return "install"
    if menu_result == 1 or menu_result == "s":
        return "skip_all"
    if menu_result == 3 or menu_result == "q":
        return "quit"
    if menu_result == 2 or menu_result == "r":
        _print_dependency_repair_commands(groups, repair_commands)
    try:
        answer = input(
            f"{key('I')} install all, {key('S')} skip affected models, {key('R')} show repair commands, {key('Q')} quit batch: "
        ).strip().lower()
    except EOFError:
        print("Skipping dependency installs before processing: input closed before confirmation.")
        return "skip_all"
    if answer in {"", "i", "install", "install all", "y", "yes"}:
        return "install"
    if answer in {"s", "skip", "skip affected", "skip affected models", "n", "no"}:
        return "skip_all"
    if answer in {"r", "repair", "show repair", "show repair commands"}:
        _print_dependency_repair_commands(groups, repair_commands)
        return _dependency_install_batch_confirmation(groups, repair_commands)
    if answer in {"q", "quit", "exit"}:
        return "quit"
    print("Unrecognized dependency install choice; skipping affected models so the rest of the batch can continue.")
    return "skip_all"


def _print_dependency_repair_commands(groups: list[str], repair_commands: dict[str, str] | None = None) -> None:
    commands = repair_commands or {}
    for group in groups:
        print(f"Manual repair command for {group}: {commands.get(group) or 'Run setup.bat --doctor for a repair command.'}")


def _repair_native_backend_preflights(candidates: list[ModelCandidate], config: dict) -> list[ModelCandidate]:
    kept: list[ModelCandidate] = []
    for candidate in candidates:
        adapter = adapter_for(candidate)
        preflight = getattr(adapter, "preflight_native_load", None)
        if preflight is None:
            kept.append(candidate)
            continue
        error = preflight(candidate, runtime_config_for_candidate(config))
        if not error:
            kept.append(candidate)
            continue
        print(f"{candidate.display_name} native backend preflight failed before model run:")
        print(f"  {error}")
        if candidate.adapter_name in {"generic_onnx_manifest", "granite_onnx_ar", "granite_onnx_nar"}:
            repaired_error = _repair_onnx_native_stack(candidate, config, error)
            if repaired_error:
                print(f"ONNX Runtime repair did not make this model loadable: {repaired_error}")
                continue
            kept.append(candidate)
            continue
        if candidate.adapter_name != "faster_whisper":
            print("Skipping this model so the app stays running.")
            continue
        if not config.get("dependency_install", {}).get("auto_install_missing_runtime_dependencies", True):
            print("Automatic dependency repair is disabled in config.json. Skipping this model.")
            continue
        decision = _dependency_install_confirmation("faster_whisper", recovery_command_for_faster_whisper_compatibility(config))
        if decision != "install":
            print("Skipped faster-whisper compatibility repair. Skipping this model.")
            continue
        repaired_error = _repair_faster_whisper_native_stack(candidate, config, error)
        if repaired_error:
            print(f"faster-whisper compatibility repair did not find a working native stack: {repaired_error}")
            continue
        kept.append(candidate)
    return kept


def _repair_onnx_native_stack(candidate: ModelCandidate, config: dict, initial_error: str) -> str:
    if not config.get("dependency_install", {}).get("auto_install_missing_runtime_dependencies", True):
        return "Automatic dependency repair is disabled in config.json."
    from .dependency_manager import install_group_for_config, recovery_command_for_config

    decision = _dependency_install_confirmation("onnx", recovery_command_for_config("onnx", config))
    if decision != "install":
        return "Skipped ONNX Runtime dependency repair."
    project_root = Path(__file__).resolve().parent.parent
    log_path = Path(_dependency_install_log_path(config, "onnx_native_preflight"))
    try:
        install_decision = install_group_for_config("onnx", project_root, config, log_path=log_path)
    except Exception as exc:
        return f"ONNX Runtime repair failed: {exc}; initial error: {initial_error}; install log: {log_path}"
    adapter = adapter_for(candidate)
    preflight = getattr(adapter, "preflight_native_load", None)
    if preflight is None:
        return ""
    error = preflight(candidate, runtime_config_for_candidate(config))
    if not error:
        repaired = install_decision.get("provider_compatibility_repair") if install_decision else ""
        if repaired:
            print(f"ONNX Runtime provider compatibility repair selected {repaired}.")
        return ""
    return error


def recovery_command_for_faster_whisper_compatibility(config: dict) -> str:
    project_root = Path(__file__).resolve().parent.parent
    requirement_file = project_root / "requirements" / "faster_whisper.txt"
    return f'"{sys.executable}" -m pip install --upgrade --force-reinstall -r "{requirement_file}"'


def _looks_like_missing_visual_cpp_runtime(error: str) -> bool:
    message = str(error or "").lower()
    return any(
        marker in message
        for marker in [
            "vcruntime",
            "msvcp",
            "api-ms-win-crt",
            "dll load failed",
            "dynamic link library",
            "the specified module could not be found",
        ]
    )


def _visual_cpp_repair_command() -> list[str]:
    from .dependency_manager import VC_REDIST_PACKAGE_ID

    return [
        "winget",
        "install",
        "-e",
        "--id",
        VC_REDIST_PACKAGE_ID,
        "--accept-package-agreements",
        "--accept-source-agreements",
    ]


def _faster_whisper_ctranslate2_spec(project_root: Path) -> str:
    try:
        from packaging.requirements import Requirement
    except Exception:
        return ""
    requirement_file = project_root / "requirements" / "faster_whisper.txt"
    if not requirement_file.exists():
        return ""
    for raw in requirement_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
        except Exception:
            continue
        if requirement.name.lower() == "ctranslate2":
            return str(requirement.specifier)
    return ""


def _installed_distribution_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _ctranslate2_version_allowed(version: str, specifier) -> bool:
    if specifier is None:
        return True
    try:
        return version in specifier
    except Exception:
        return False


def _discover_ctranslate2_candidate_versions(config: dict, project_root: Path, log) -> list[str]:
    spec_text = _faster_whisper_ctranslate2_spec(project_root)
    try:
        from packaging.specifiers import SpecifierSet
    except Exception:
        SpecifierSet = None  # type: ignore[assignment]
    specifier = SpecifierSet(spec_text) if SpecifierSet and spec_text else None
    configured = config.get("dependency_install", {}).get("ctranslate2_compatibility_versions")
    if configured:
        return [version for version in (str(item) for item in configured) if _ctranslate2_version_allowed(version, specifier)]
    command = [sys.executable, "-m", "pip", "index", "versions", "ctranslate2"]
    log.write(f"\n> {' '.join(command)}\n")
    try:
        completed = subprocess.run(command, cwd=str(project_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.write(f"pip index failed: {exc}\n")
        completed = None
    versions: list[str] = []
    if completed is not None:
        log.write(completed.stdout or "")
        for line in (completed.stdout or "").splitlines():
            if "Available versions:" not in line:
                continue
            raw_versions = line.split("Available versions:", 1)[1].split(",")
            for raw_version in raw_versions:
                version = raw_version.strip()
                if version and _ctranslate2_version_allowed(version, specifier):
                    versions.append(version)
    installed = _installed_distribution_version("ctranslate2")
    if installed and _ctranslate2_version_allowed(installed, specifier) and installed not in versions:
        versions.insert(0, installed)
    return versions


def _repair_faster_whisper_native_stack(candidate: ModelCandidate, config: dict, initial_error: str) -> str:
    from .adapters.faster_whisper_asr import probe_faster_whisper_load, faster_whisper_runtime_choices
    from .dependency_manager import visual_cpp_redistributable_status

    project_root = Path(__file__).resolve().parent.parent
    _plan, device, _compute_type, effective_compute_type, _warnings = faster_whisper_runtime_choices(candidate, runtime_config_for_candidate(config))
    last_error = initial_error
    log_path = Path(_dependency_install_log_path(config, "faster_whisper_native_compatibility"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"Initial faster-whisper native preflight error: {initial_error}\n")
        vc_status = visual_cpp_redistributable_status()
        if _looks_like_missing_visual_cpp_runtime(initial_error) and not vc_status.get("installed", False):
            vc_command = _visual_cpp_repair_command()
            log.write("Visual C++ Redistributable was not detected and the native error looks like a missing DLL/runtime.\n")
            log.write(f"\n> {' '.join(vc_command)}\n")
            print("Trying Microsoft Visual C++ Redistributable repair before replacing CTranslate2 packages...")
            try:
                completed = subprocess.run(vc_command, cwd=str(project_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
            except (OSError, subprocess.TimeoutExpired) as exc:
                log.write(f"Visual C++ Redistributable repair command failed to launch: {exc}\n")
                last_error = f"Visual C++ Redistributable repair command failed to launch: {exc}; initial error: {initial_error}"
            else:
                log.write(completed.stdout or "")
                if completed.returncode == 0:
                    probe_error = probe_faster_whisper_load(candidate.path, device, effective_compute_type)
                    log.write(f"probe after Visual C++ Redistributable repair: {'pass' if not probe_error else probe_error}\n")
                    if not probe_error:
                        print("Visual C++ Redistributable repair made faster-whisper native load pass.")
                        return ""
                    last_error = probe_error
                else:
                    last_error = f"Visual C++ Redistributable repair failed; see {log_path}"
        broad_command = [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "-r", str(project_root / "requirements" / "faster_whisper.txt")]
        log.write(f"\n> {' '.join(broad_command)}\n")
        print("Trying the latest package set allowed by requirements/faster_whisper.txt...")
        completed = subprocess.run(broad_command, cwd=str(project_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write(completed.stdout or "")
        if completed.returncode == 0:
            probe_error = probe_faster_whisper_load(candidate.path, device, effective_compute_type)
            log.write(f"probe requirement set: {'pass' if not probe_error else probe_error}\n")
            if not probe_error:
                print("faster-whisper requirement set passed the native load probe.")
                return ""
            last_error = probe_error
        else:
            last_error = f"pip install requirements/faster_whisper.txt failed; see {log_path}"
        versions = _discover_ctranslate2_candidate_versions(config, project_root, log)
        if not versions:
            return f"{last_error}; no CTranslate2 candidates could be discovered from pip index or the installed environment"
        for version in versions:
            command = [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", f"ctranslate2=={version}", "-r", str(project_root / "requirements" / "faster_whisper.txt")]
            log.write(f"\n> {' '.join(command)}\n")
            print(f"Trying discovered CTranslate2 candidate {version}...")
            completed = subprocess.run(command, cwd=str(project_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.write(completed.stdout or "")
            if completed.returncode != 0:
                last_error = f"pip install ctranslate2=={version} failed; see {log_path}"
                continue
            probe_error = probe_faster_whisper_load(candidate.path, device, effective_compute_type)
            log.write(f"probe ctranslate2=={version}: {'pass' if not probe_error else probe_error}\n")
            if not probe_error:
                print(f"CTranslate2 {version} passed the native load probe.")
                return ""
            last_error = probe_error
    return last_error


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


def classify_model_failure(exc: BaseException, candidate: ModelCandidate) -> tuple[list[str], list[str]]:
    message = str(exc).lower()
    family = candidate.family_name or candidate.display_name
    causes: list[str] = []
    actions: list[str] = []
    if isinstance(exc, ImportError) or "no module named" in message or "modulenotfounderror" in message:
        causes.append(f"The optional runtime package for {family} is not installed or is installed in the wrong environment.")
        actions.append("Run the dependency install prompt for this model, or run setup.bat --doctor for the exact repair command.")
    if "cuda" in message or "cudnn" in message or "cublas" in message or "gpu" in message:
        causes.append("The selected GPU provider failed, is missing CUDA/cuDNN runtime files, or is incompatible with this model/backend.")
        actions.append("Retry with CPU when available, or install the GPU runtime shown by setup.bat --doctor.")
    if "onnx" in message or "executionprovider" in message or "provider" in message:
        causes.append("The ONNX provider, model sidecar files, or provider-specific runtime is missing or incompatible.")
        actions.append("Confirm the model folder includes every required ONNX sidecar file and use the provider repair command from doctor.")
    if "out of memory" in message or "oom" in message or "memory" in message:
        causes.append("The model may be too large for available RAM/VRAM or the selected precision/provider.")
        actions.append("Try a smaller model, lower-precision model, CPU mode, or fewer simultaneous models.")
    if "file" in message or "path" in message or "not found" in message or "missing" in message:
        causes.append("The model folder may be incomplete, moved, locked, or missing required files.")
        actions.append("Use the model scanner or Hugging Face downloader repair prompt to restore missing same-package files.")
    if not causes:
        causes = [
            "The model runtime failed while loading or transcribing this file.",
            "The model package, provider, dependency group, or selected runtime option may be incompatible.",
        ]
    if not actions:
        actions = [
            "Check the model folder and dependency group shown in this report.",
            "Run setup.bat --doctor for runtime status and repair commands.",
            "Try another runnable ASR model so the batch can continue.",
        ]
    return causes, actions


def build_model_failure_error(
    *,
    stage: str,
    candidate: ModelCandidate,
    exc: BaseException,
    config: dict,
    log_path: Path | None,
) -> dict:
    from .dependency_manager import recovery_command_for_config

    dependency_groups = list(candidate.dependency_groups or [])
    repair_command = ""
    if dependency_groups:
        try:
            repair_command = recovery_command_for_config(dependency_groups[0], config)
        except Exception:
            repair_command = "Run setup.bat --doctor for a repair command."
    likely_causes, next_actions = classify_model_failure(exc, candidate)
    return {
        "status": "model_failed",
        "stage": stage,
        "model_id": candidate.candidate_id,
        "model_name": candidate.display_name,
        "model_path": str(candidate.path),
        "error_type": type(exc).__name__,
        "message": str(exc) or type(exc).__name__,
        "likely_causes": likely_causes,
        "next_actions": next_actions,
        "dependency_group": ", ".join(dependency_groups),
        "provider_requested": str(config.get("runtime", {}).get("provider", "auto")),
        "provider_actual": "",
        "repair_command": repair_command,
        "log_path": str(log_path) if log_path else "",
        "traceback": traceback.format_exc(),
    }


def process_file_with_candidates(
    source: Path,
    candidates: list[ModelCandidate],
    config: dict,
    unsupported_models: list[ModelCandidate] | None = None,
    reference_llm: ModelCandidate | None = None,
    file_progress: tuple[int, int] | None = None,
) -> Path | None:
    from .benchmark import peak_vram_sample, process_memory_mb, reset_peak_vram, timer
    from .media import audio_duration_seconds, prepare_audio
    from .results_writer import build_failed_file_results, build_results, write_all_reports

    source = source.resolve()
    temp_dir = folder_config(config, "temp", "temp_folder")
    output_dir = folder_config(config, "output")
    logging.info("Processing %s", source)
    progress_prefix = f"[File {file_progress[0]}/{file_progress[1]}] " if file_progress else ""
    print(f"{progress_prefix}Processing {source.name}")
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
        for model_index, candidate in enumerate(candidates, 1):
            logging.info("Running %s", candidate.candidate_id)
            print(f"{progress_prefix}[Model {model_index}/{len(candidates)}] Running {candidate.display_name}")
            adapter = None
            try:
                adapter = adapter_for(candidate)
                load_started = time.perf_counter()
                reset_peak_vram()
                adapter.load(candidate, runtime_config_for_candidate(config))
                model_load_seconds = time.perf_counter() - load_started
                result = adapter.transcribe_chunks(chunks, chunk_metadata)
                result.metrics["model_load_seconds"] = model_load_seconds
                result.metrics["total_wall_seconds"] = model_load_seconds + float(result.metrics.get("inference_seconds", 0))
                audio_seconds_metric = float(result.metrics.get("audio_seconds", audio_seconds))
                result.metrics["audio_seconds_per_wall_second"] = audio_seconds_metric / max(0.001, float(result.metrics["total_wall_seconds"]))
                result.metrics.update(peak_vram_sample())
                print(f"{progress_prefix}[Model {model_index}/{len(candidates)}] Finished {candidate.display_name} in {result.metrics['total_wall_seconds']:.2f} seconds")
            except Exception as exc:
                logging.error("%s failed", candidate.candidate_id)
                logging.exception("Model failed")
                print(f"{progress_prefix}[Model {model_index}/{len(candidates)}] Failed {candidate.display_name}: {exc}")
                result = ModelRunResult(
                    candidate=candidate,
                    transcript_chunks=[],
                    metrics={
                        "provider": str(config["runtime"].get("provider", "auto")),
                        "audio_seconds": audio_seconds,
                        "chunk_count": len(chunks),
                        "media_normalization_seconds": media_seconds,
                        "peak_process_memory_mb": process_memory_mb(),
                        **peak_vram_sample(),
                    },
                    errors=[
                        build_model_failure_error(
                            stage="model_load_or_inference",
                            candidate=candidate,
                            exc=exc,
                            config=config,
                            log_path=latest_log_path(config),
                        )
                    ],
                )
            finally:
                if adapter is not None:
                    try:
                        adapter.unload()
                    except Exception as exc:
                        logging.exception("Adapter unload failed")
                        result.errors.append(
                            build_model_failure_error(
                                stage="adapter_unload",
                                candidate=candidate,
                                exc=exc,
                                config=config,
                                log_path=latest_log_path(config),
                            )
                        )
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
                    print(f"{progress_prefix}[Reference LLM] Generating corrected reference with {reference_llm.display_name}")
                    results["local_llm_reference_attempt"] = ref_adapter.generate_reference(reference_llm, runtime_config_for_candidate(config), results)
                    print(f"{progress_prefix}[Reference LLM] Finished corrected reference attempt")
            except Exception as exc:
                print(f"{progress_prefix}[Reference LLM] Failed corrected reference attempt: {exc}")
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
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--repair-plan", action="store_true")
    parser.add_argument("--repair-all-safe", action="store_true")
    parser.add_argument("--validate-real-smoke", action="store_true")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--full-real-smoke", action="store_true")
    parser.add_argument("--first-run", action="store_true")
    parser.add_argument("--first-run-smoke", action="store_true")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--download-model-first", action="store_true")
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
    except Exception as exc:
        _handle_fatal_error(exc, args)


def _logs_dir_from_args(args: argparse.Namespace) -> Path:
    config_path = Path(getattr(args, "config", "config.json"))
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        folders = data.get("folders", {}) if isinstance(data, dict) else {}
        if isinstance(folders, dict) and folders.get("logs"):
            return Path(str(folders["logs"]))
        advanced = data.get("advanced", {}) if isinstance(data, dict) else {}
        if isinstance(advanced, dict) and advanced.get("logs_folder"):
            return Path(str(advanced["logs_folder"]))
    except Exception:
        pass
    return Path("Logs")


def _handle_fatal_error(exc: Exception, args: argparse.Namespace) -> None:
    logs_dir = _logs_dir_from_args(args)
    logs_dir.mkdir(parents=True, exist_ok=True)
    crash_path = logs_dir / f"crash_{time.strftime('%Y%m%d_%H%M%S')}.log"
    crash_path.write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
    print()
    print("Easy ASR Bench hit an unexpected error and stopped before it could finish.")
    print(f"Problem: {type(exc).__name__}: {exc}")
    print(f"Crash log: {crash_path}")
    print("Run setup.bat --doctor --strict for environment diagnostics.")
    print("Report bugs at: https://github.com/rollingedit/Easy-ASR-Bench/issues/new/choose")


def _main(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    from .update_check import check_for_updates_from_config

    check_for_updates_from_config(config, context="run")
    if args.doctor:
        from .doctor import run_doctor

        raise SystemExit(
            run_doctor(
                Path(args.config),
                strict=bool(args.strict),
                json_output=bool(args.json),
                repair_plan_output=bool(args.repair_plan),
                repair_all_safe=bool(args.repair_all_safe),
                validate_real_smoke=bool(args.validate_real_smoke),
                install_deps=bool(args.install_deps),
                allow_downloads=bool(args.allow_downloads),
                no_network=bool(args.no_network),
                full_real_smoke=bool(args.full_real_smoke),
            )
        )
    models_root = folder_config(config, "models")
    if args.first_run_smoke:
        from .first_run import build_first_run_smoke_report

        print(json.dumps(build_first_run_smoke_report(config), indent=2))
        return
    if args.open_models:
        open_folder(models_root)
        return
    if args.open_input:
        open_folder(folder_config(config, "input"))
        return
    if args.download_model:
        destination = download_hf_model_interactive(models_root)
        if destination is None:
            return
        print(f"Model package saved. Rescanning models now: {destination}")
        args.interactive = True
    if args.first_run:
        from .first_run import run_first_run_wizard

        initial_action = "paste_hf" if args.download_model_first else None
        if not run_first_run_wizard(config, initial_action=initial_action):
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
        print("Core runtime ready. Optional model runtimes install only when selected.")
        selected, reference_llm = choose_candidates(runnable, unsupported, config, Path(args.config), models_root)
    else:
        if args.paths:
            saved_selected, saved_reference_llm, saved_errors = resolve_last_run_selection(runnable, unsupported, config)
            if saved_selected:
                selected = saved_selected
                reference_llm = saved_reference_llm
                print("Using saved last-run model selection: " + ", ".join(candidate.display_name for candidate in selected))
            elif isinstance(config.get("last_run_selection"), dict):
                print("Saved last-run model selection is stale: " + "; ".join(saved_errors))
                print("Run interactively once to choose models again, or remove last_run_selection from config.json.")
                return
            else:
                print("No saved last-run model selection found; using all runnable ASR models. Run interactively once to save a repeatable selection.")
                selected = [candidate for candidate in runnable if candidate.category == "asr"]
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
                for file_index, file_path in enumerate(files, 1):
                    progress = (file_index, len(files)) if args.once else None
                    output = process_file_with_candidates(file_path, selected, config, unsupported, reference_llm, file_progress=progress)
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
        for file_index, file_path in enumerate(files, 1):
            from .queue_manager import QueueItem
            from .utils import file_key

            state.upsert(QueueItem(str(file_path.resolve()), "", file_key(file_path)))
            output = process_file_with_candidates(file_path, selected, config, unsupported, reference_llm, file_progress=(file_index, len(files)))
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
    print(f"Open final results: {output_dir / 'final_results.html'}")


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
