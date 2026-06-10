from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


MEDIA_EXTENSIONS = [
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
]


DEFAULT_CONFIG: dict[str, Any] = {
    "app": {
        "version": "0.4.0",
        "version_channel": "prerelease",
        "check_for_updates_on_setup": True,
        "check_for_updates_on_run": False,
    },
    "folders": {
        "models": "Models",
        "input": "Input",
        "output": "Output",
        "temp": "Temp",
        "logs": "Logs",
        "cache": "Cache",
    },
    "runtime": {
        "execution": "sequential",
        "provider": "cpu",
        "prefer_gpu": False,
        "fallback_to_cpu": True,
        "unload_between_models": True,
        "continue_after_model_error": True,
        "cpu_threads": 0,
        "gpu_device_id": 0,
    },
    "dependency_install": {
        "auto_install_missing_runtime_dependencies": True,
        "prefer_cpu_safe_defaults": True,
        "allow_cuda_install": False,
        "allow_accelerator_install": False,
    },
    "model_scan": {
        "recursive": True,
    },
    "security": {
        "trust_remote_code": False,
        "allow_model_folder_scripts": False,
        "allow_pickle_or_pt_files": False,
        "allow_known_official_whisper_pt": False,
        "scan_only_safe_formats_by_default": True,
        "allow_manifest_custom_python": False,
    },
    "input": {
        "recursive_folders": True,
        "file_stability_wait_seconds": 5,
        "extensions": MEDIA_EXTENSIONS,
    },
    "transcription": {
        "task": "transcribe",
        "ar_prompt": "transcribe the speech with proper punctuation and capitalization.",
        "ar_max_new_tokens": 1024,
        "language": "auto",
        "temperature": 0.0,
    },
    "whisper": {
        "task": "transcribe",
        "language": "auto",
        "return_timestamps": False,
        "chunk_length_s": 30,
        "stride_length_s": 5,
        "batch_size": 1,
    },
    "chunking": {
        "mode": "auto_smart",
        "target_chunk_seconds": 240,
        "hard_max_chunk_seconds": 480,
        "boundary_search_seconds": 20,
        "min_silence_ms": 350,
        "silence_threshold_db": -35,
        "rms_fallback_window_ms": 250,
        "allow_overlap": False,
        "fallback": "lowest_rms_boundary",
    },
    "benchmark": {
        "measure_peak_ram": True,
    },
    "reports": {
        "write_txt": True,
        "write_json": True,
        "write_html": True,
        "embed_results_json_in_html": True,
        "include_llm_reference_instructions": True,
    },
    "llm_reference": {
        "custom_model_paths": [],
        "auto_scan_saved_paths": True,
        "manual_external_llm_guide": True,
    },
    "llama_cpp": {
        "context_size": 10240,
        "timeout_seconds": 600,
        "mtmd_cli_path": "",
    },
    "advanced": {
        "keep_temp_wavs": False,
    },
}


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        save_default_config(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return deep_merge(DEFAULT_CONFIG, data)


def save_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def save_default_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(DEFAULT_CONFIG, handle, indent=2)
        handle.write("\n")


def selected_variants(config: dict[str, Any]) -> list[str]:
    del config
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--path", default="config.json")
    args = parser.parse_args()
    path = Path(args.path)
    if args.init:
        if not path.exists():
            save_default_config(path)
            print(f"Created {path}")
        else:
            print(f"{path} already exists")
        return
    print(json.dumps(load_config(path), indent=2))


if __name__ == "__main__":
    main()
