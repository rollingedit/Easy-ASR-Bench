from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from check_release_version_coherence import validate as validate_version_coherence
except ModuleNotFoundError:
    from scripts.check_release_version_coherence import validate as validate_version_coherence


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def current_commit() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True)
    return completed.stdout.strip()


def run_check(name: str, command: list[str]) -> dict:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {
        "name": name,
        "status": "pass" if completed.returncode == 0 else "fail",
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def verify_assets(tag: str) -> tuple[bool, dict[str, str]]:
    checksums = json.loads((ROOT / "installer" / "checksums.json").read_text(encoding="utf-8"))
    files = checksums.get("files", {})
    paths = {}
    for name in files:
        if name.endswith(".zip"):
            paths[name] = ROOT / "dist" / name
        elif name in {"manifest.json", "checksums.json"}:
            paths[name] = ROOT / "installer" / name
        elif name == "install.ps1" and not (ROOT / name).exists():
            paths[name] = ROOT / "installer" / "install.ps1"
        else:
            paths[name] = ROOT / name
    actual = {name: sha256(path) for name, path in paths.items()}
    return actual == files and checksums.get("version") == tag.lstrip("v"), actual


def manual_rows_from_matrix(matrix: dict) -> list[dict]:
    rows: list[dict] = []
    for key, value in matrix.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                rows.append({"id": nested_key, "status": nested_value})
        else:
            rows.append({"id": key, "status": value})
    return rows


def write_smoke(tag: str, output: Path, commit: str | None = None) -> None:
    tag = tag if tag.startswith("v") else f"v{tag}"
    zip_path = ROOT / "dist" / f"Easy-ASR-Bench-{tag}-win.zip"
    if not zip_path.exists():
        raise SystemExit(f"Release ZIP is missing: {zip_path}")

    validate_version_coherence(tag)
    assets_verified, asset_hashes = verify_assets(tag)
    if not assets_verified:
        raise SystemExit("Release asset hashes do not match installer/checksums.json")

    checks = [
        run_check("release_file_validation", [sys.executable, "scripts/validate_release_files.py"]),
        run_check("repo_physical_file_validation", [sys.executable, "scripts/validate_physical_files.py", "--repo", "."]),
        run_check("zip_physical_file_validation", [sys.executable, "scripts/validate_physical_files.py", "--zip", str(zip_path)]),
        run_check("version_coherence", [sys.executable, "scripts/check_release_version_coherence.py", "--tag", tag]),
    ]
    if any(check["status"] != "pass" for check in checks):
        raise SystemExit("One or more release smoke checks failed")

    manual_matrix = {
        "win11_clean_no_python_setup": "not_run",
        "win10_existing_python_setup": "not_run",
        "install_path_with_spaces": "not_run",
        "setup_verify_release_bad_checksum": "not_run",
        "setup_double_click_equivalent": "not_run",
        "setup_dry_run_verify_release": "not_run",
        "setup_doctor_strict": "not_run",
        "update_preserves_user_data": "not_run",
        "repair_broken_venv": "not_run",
        "uninstall_preserve_user_data": "not_run",
        "destructive_uninstall_requires_phrase": "not_run",
        "bad_checksum_fails_before_execution": "not_run",
        "tampered_installer_fails_before_execution": "not_run",
        "interrupted_download_rollback": "not_run",
        "broken_venv_repair": "not_run",
        "empty_models_folder": "not_run",
        "empty_models": "not_run",
        "nested_models_folders": "not_run",
        "nested_models_scan": "not_run",
        "wav_mp3_mp4_no_audio_corrupt_media": "not_run",
        "wav_mp3_mp4_media": "not_run",
        "corrupt_media_readable_error": "not_run",
        "no_audio_video_readable_error": "not_run",
        "compare_html_offline_large_transcript": "not_run",
        "compare_html_offline": "not_run",
        "batch_continues_after_one_model_or_chunk_fails": "not_run",
        "one_model_failure_continues": "not_run",
        "one_chunk_failure_continues": "not_run",
        "llm_reference_json_import": "not_run",
        "dependency_install_declined": "not_run",
        "provider_smoke": {
            "cpu_model_smoke": "not_run",
            "nvidia_cuda_torch_onnx_faster_whisper_llama": "not_run",
            "amd_directml_onnx_smoke": "not_run",
            "intel_directml_onnx_smoke": "not_run",
            "intel_openvino_onnx_smoke": "not_run",
            "vulkan_runtime_no_sdk": "not_run",
            "vulkan_runtime_with_sdk": "not_run",
        },
        "model_smoke": {
            "hf_safetensors_asr": "not_run",
            "hf_whisper_safetensors": "not_run",
            "hf_whisper_safetensors_cpu": "not_run",
            "sharded_safetensors_index": "not_run",
            "faster_whisper_ctranslate2": "not_run",
            "faster_whisper_cpu": "not_run",
            "faster_whisper_cuda_unavailable_cpu_fallback": "not_run",
            "whisper_cpp_ggml": "not_run",
            "openai_whisper_pt_checksum_verified": "not_run",
            "openai_whisper_pt_unknown_blocked": "not_run",
            "openai_pt_unverified_blocked": "not_run",
            "generic_onnx_ctc_manifest_v1": "not_run",
            "generic_onnx_manifest_cpu": "not_run",
            "generic_onnx_without_manifest_rejected": "not_run",
            "multi_file_onnx_ar_nar": "not_run",
            "audio_asr_gguf_mmproj": "not_run",
            "gguf_asr_mmproj_pair": "not_run",
            "incomplete_audio_asr_gguf_mmproj_rejected": "not_run",
            "gguf_reference_llm": "not_run",
            "gguf_text_llm_reference_only": "not_run",
            "standalone_safetensors_incomplete": "not_run",
            "hf_text_llm_safetensors_unsupported": "not_run",
            "known_unsupported_asr_families_explained": "not_run",
        },
    }
    smoke = {
        "schema": "easy_asr_bench.release_smoke.v2",
        "tag": tag,
        "commit": commit or current_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "asset_hashes_verified": assets_verified,
        "asset_hashes": asset_hashes,
        "checks": checks,
        "manual_matrix": manual_matrix,
        "manual_rows": manual_rows_from_matrix(manual_matrix),
        "notes": [
            "This artifact records automated release-candidate validation only.",
            "Rows marked not_run require manual Windows/hardware/model smoke testing before claiming production readiness.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_smoke(args.tag, args.output, args.commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
