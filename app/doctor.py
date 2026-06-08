from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .dependency_manager import acceleration_install_decision, cuda_diagnostics, dependency_status
from .hf_model_downloader import execute_persisted_missing_file_repair_plan
from .repair_plan import build_repair_plan, execute_repair_plan
from .version import RELEASE_CHANNEL, RELEASE_COMMIT, TAG


DEFAULT_REAL_SMOKE_ROWS = ["setup_repair_all_safe", "cpu_model_smoke", "compare_html_offline"]
DEFAULT_FULL_REAL_SMOKE_ROWS = [
    "setup_repair_all_safe",
    "cpu_model_smoke",
    "compare_html_offline",
    "real_tiny_faster_whisper_smollm_grading",
    "real_public_media_faster_whisper_smollm_grading",
    "real_public_media_openai_whisper_pt_smollm_grading",
    "real_public_video_openai_whisper_pt_smollm_grading",
    "real_public_media_whisper_cpp_ggml_smollm_grading",
    "real_public_media_hf_whisper_safetensors_smollm_grading_cpu",
    "real_public_media_generic_onnx_ctc_smollm_grading_cpu",
    "real_public_media_gguf_asr_mmproj_smollm_grading",
    "hf_safetensors_asr_quality_smollm_grading_cpu",
    "same_media_multi_model_smollm_benchmark",
    "same_media_multi_model_smollm_benchmark_directml",
]


def build_doctor_report(config_path: Path) -> dict:
    config = load_config(config_path)
    folders = config["folders"]
    for folder in folders.values():
        Path(folder).mkdir(parents=True, exist_ok=True)
    status = dependency_status(config)
    cuda = cuda_diagnostics()
    return {
        "schema": "easy_asr_bench.doctor.v1",
        "version": TAG,
        "release_channel": RELEASE_CHANNEL,
        "release_commit": RELEASE_COMMIT,
        "config_version": config.get("app", {}).get("version"),
        "config_channel": config.get("app", {}).get("version_channel"),
        "dependency_status": status,
        "cuda_provider_checks": cuda,
        "folders": {key: str(value) for key, value in folders.items()},
        "repair_plan": build_repair_plan(config, status=status),
    }


def run_real_smoke_validation(
    config_path: Path,
    *,
    install_deps: bool = False,
    allow_downloads: bool = False,
    no_network: bool = False,
    full_real_smoke: bool = False,
) -> dict:
    config = load_config(config_path)
    project_root = Path(__file__).resolve().parent.parent
    runtime_validation = config.get("runtime_validation", {})
    if full_real_smoke:
        rows = list(runtime_validation.get("full_smoke_rows") or DEFAULT_FULL_REAL_SMOKE_ROWS)
        smoke_profile = "full"
    else:
        rows = list(runtime_validation.get("smoke_rows") or DEFAULT_REAL_SMOKE_ROWS)
        smoke_profile = "quick"
    workdir = project_root / "Temp" / "doctor_real_smoke"
    repair = execute_repair_plan(config, project_root=project_root)
    results = []
    effective_allow_downloads = allow_downloads and not no_network
    for row_id in rows:
        command = [
            sys.executable,
            "qa/runtime_matrix/run_row.py",
            "--row",
            row_id,
            "--workdir",
            str(workdir),
        ]
        if install_deps:
            command.append("--install-deps")
        if effective_allow_downloads:
            command.append("--allow-downloads")
        completed = subprocess.run(command, cwd=project_root, text=True, capture_output=True, timeout=3600)
        row_path = workdir / row_id / "row.json"
        payload = {}
        if row_path.exists():
            try:
                payload = json.loads(row_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                payload = {"status": "invalid_json", "read_error": str(exc)}
        results.append(
            {
                "id": row_id,
                "command": command,
                "exit_code": completed.returncode,
                "status": payload.get("status", "missing"),
                "row_json": str(row_path),
                "summary": payload.get("summary", ""),
                "block_reason": payload.get("block_reason", ""),
                "external_requirement": payload.get("external_requirement", ""),
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        )
    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema": "easy_asr_bench.real_smoke_validation.v1",
        "mode": "validate_real_smoke",
        "version": TAG,
        "release_commit": RELEASE_COMMIT,
        "config": str(config_path),
        "workdir": str(workdir),
        "install_deps": install_deps,
        "allow_downloads": effective_allow_downloads,
        "requested_allow_downloads": allow_downloads,
        "no_network": no_network,
        "network_policy": "no_network" if no_network else ("allow_downloads" if effective_allow_downloads else "offline_default"),
        "smoke_profile": smoke_profile,
        "full_real_smoke": full_real_smoke,
        "repair_all_safe": repair,
        "rows": results,
        "summary": {
            "row_count": len(results),
            "status_counts": status_counts,
            "failed": sum(1 for result in results if result["exit_code"] != 0 or result["status"] == "fail"),
            "blocked": status_counts.get("blocked", 0),
            "passed": status_counts.get("pass", 0),
        },
    }


def run_model_layout_repair_sweep(config_path: Path, *, allow_downloads: bool = False, no_network: bool = False) -> dict:
    config = load_config(config_path)
    models_root = Path(config.get("folders", {}).get("models", "Models"))
    if not models_root.is_absolute():
        models_root = Path.cwd() / models_root
    effective_allow_downloads = allow_downloads and not no_network
    plan_paths = sorted(models_root.rglob("hf_model_layout_repair_plan.json")) if models_root.exists() else []
    executions = [
        execute_persisted_missing_file_repair_plan(path, allow_downloads=effective_allow_downloads, print_func=lambda _text: None)
        for path in plan_paths
    ]
    return {
        "schema": "easy_asr_bench.model_layout_repair_sweep.v1",
        "mode": "repair_model_layouts",
        "version": TAG,
        "release_commit": RELEASE_COMMIT,
        "config": str(config_path),
        "models_root": str(models_root),
        "allow_downloads": effective_allow_downloads,
        "requested_allow_downloads": allow_downloads,
        "no_network": no_network,
        "plan_count": len(plan_paths),
        "plan_paths": [str(path) for path in plan_paths],
        "executions": executions,
        "summary": {
            "plan_count": len(plan_paths),
            "repaired": sum(item.get("summary", {}).get("repaired", 0) for item in executions),
            "blocked": sum(item.get("summary", {}).get("blocked", 0) for item in executions),
            "failed": sum(item.get("summary", {}).get("failed", 0) for item in executions),
            "downloaded_files": sum(item.get("summary", {}).get("downloaded_files", 0) for item in executions),
        },
    }


def run_doctor(
    config_path: Path,
    strict: bool = False,
    json_output: bool = False,
    repair_plan_output: bool = False,
    repair_all_safe: bool = False,
    validate_real_smoke: bool = False,
    repair_model_layouts: bool = False,
    install_deps: bool = False,
    allow_downloads: bool = False,
    no_network: bool = False,
    full_real_smoke: bool = False,
) -> int:
    config = load_config(config_path)
    report = build_doctor_report(config_path)
    status = report["dependency_status"]
    if repair_all_safe:
        print(json.dumps(execute_repair_plan(config, status=status), indent=2))
        return 0 if (status["core"]["available"] or not strict) else 1
    if validate_real_smoke:
        report = run_real_smoke_validation(
            config_path,
            install_deps=install_deps,
            allow_downloads=allow_downloads,
            no_network=no_network,
            full_real_smoke=full_real_smoke,
        )
        print(json.dumps(report, indent=2))
        if strict and report["summary"]["failed"]:
            return 1
        return 0 if (status["core"]["available"] or not strict) else 1
    if repair_model_layouts:
        report = run_model_layout_repair_sweep(config_path, allow_downloads=allow_downloads, no_network=no_network)
        print(json.dumps(report, indent=2))
        if strict and report["summary"]["failed"]:
            return 1
        return 0 if (status["core"]["available"] or not strict) else 1
    if repair_plan_output:
        print(json.dumps(report["repair_plan"], indent=2))
        return 0 if (status["core"]["available"] or not strict) else 1
    if json_output:
        print(json.dumps(report, indent=2))
        return 0 if (status["core"]["available"] or not strict) else 1
    print("Easy ASR Bench Doctor")
    print(f"Version: {report['version']}")
    print(f"Release channel: {report['release_channel']}")
    if report.get("config_channel") and report["config_channel"] != report["release_channel"]:
        print(f"Config channel note: config.json says {report['config_channel']}, but this build reports {report['release_channel']}.")
    print(f"Release commit: {report['release_commit']}")
    print()
    for group, data in status.items():
        mark = "OK" if data["available"] else "MISSING"
        print(f"{mark:8} {group} - {data['description']}")
        if data["missing"]:
            print("         missing: " + ", ".join(data["missing"]))
            print("         repair: " + data["recovery_command"])
            acceleration_decision = acceleration_install_decision(config, group)
            if data.get("accelerator_recovery_command") and acceleration_decision["use_accelerator"]:
                print(f"         {acceleration_decision['accelerator']} repair: " + data["accelerator_recovery_command"])
            elif data.get("cuda_recovery_command"):
                print("         accelerator note: " + acceleration_decision["reason"])
    print()
    print("CUDA/provider checks:")
    cuda = report["cuda_provider_checks"]
    print(f"  NVIDIA GPU detected: {cuda['nvidia_gpu_detected']}")
    print(f"  AMD GPU detected: {cuda.get('amd_gpu_detected', False)}")
    print(f"  Intel GPU/NPU detected: {cuda.get('intel_gpu_or_npu_detected', False)}")
    print(f"  Windows GPU detected: {cuda.get('windows_gpu_detected', False)}")
    print(f"  Vulkan tooling detected: {cuda.get('vulkan_detected', False)}")
    print(f"  Vulkan SDK detected: {cuda.get('vulkan_sdk_detected', False)}")
    vc_status = cuda.get("visual_cpp_redistributable", {})
    print(f"  Visual C++ 2015-2022 Redistributable x64 detected: {vc_status.get('installed', False)}")
    if vc_status.get("version"):
        print(f"  Visual C++ Redistributable version: {vc_status['version']}")
    if vc_status.get("repair_command") and not vc_status.get("installed", False):
        print(f"  Visual C++ Redistributable repair: {vc_status['repair_command']}")
    hf_cache = cuda.get("huggingface_cache", {})
    if hf_cache:
        print(f"  Hugging Face cache dir: {hf_cache.get('cache_dir', '')}")
        print(f"  Hugging Face cache symlink support: {hf_cache.get('symlink_supported')}")
        if hf_cache.get("repair_guidance"):
            print(f"  Hugging Face cache note: {hf_cache['repair_guidance']}")
    print(f"  torch installed: {cuda['torch_installed']}")
    print(f"  torch CUDA available: {cuda['torch_cuda_available']}")
    if cuda["torch_cuda_version"]:
        print(f"  torch CUDA version: {cuda['torch_cuda_version']}")
    if cuda["torch_gpu_names"]:
        print(f"  torch GPUs: {', '.join(cuda['torch_gpu_names'])}")
    print(f"  onnxruntime installed: {cuda['onnxruntime_installed']}")
    print(f"  onnxruntime providers: {', '.join(cuda['onnxruntime_providers']) or 'none'}")
    mtmd_status = cuda.get("llama_mtmd_cli", {})
    print(f"  llama-mtmd-cli detected: {mtmd_status.get('available', False)}")
    if mtmd_status.get("path"):
        print(f"  llama-mtmd-cli path: {mtmd_status['path']}")
    if mtmd_status.get("qwen3_asr_handler_available"):
        print("  llama-cpp-python Qwen3 ASR handler detected: True")
    if mtmd_status.get("repair_command") and not mtmd_status.get("available", False):
        print(f"  llama-mtmd-cli repair: {mtmd_status['repair_command']}")
    for message in cuda["messages"]:
        print(f"  note: {message}")
    print()
    print("Folders checked:")
    for key, folder in report["folders"].items():
        print(f"  {key}: {folder}")
    return 0 if (status["core"]["available"] or not strict) else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--repair-plan", action="store_true")
    parser.add_argument("--repair-all-safe", action="store_true")
    parser.add_argument("--validate-real-smoke", action="store_true")
    parser.add_argument("--repair-model-layouts", action="store_true")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--full-real-smoke", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run_doctor(
            Path(args.config),
            strict=args.strict,
            json_output=args.json,
            repair_plan_output=args.repair_plan,
            repair_all_safe=args.repair_all_safe,
            validate_real_smoke=args.validate_real_smoke,
            repair_model_layouts=args.repair_model_layouts,
            install_deps=args.install_deps,
            allow_downloads=args.allow_downloads,
            no_network=args.no_network,
            full_real_smoke=args.full_real_smoke,
        )
    )


if __name__ == "__main__":
    main()
