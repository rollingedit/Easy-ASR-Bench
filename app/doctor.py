from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .dependency_manager import acceleration_install_decision, cuda_diagnostics, dependency_status
from .version import RELEASE_CHANNEL, RELEASE_COMMIT, TAG


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
    }


def run_doctor(config_path: Path, strict: bool = False, json_output: bool = False) -> int:
    config = load_config(config_path)
    report = build_doctor_report(config_path)
    status = report["dependency_status"]
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
    print(f"  torch installed: {cuda['torch_installed']}")
    print(f"  torch CUDA available: {cuda['torch_cuda_available']}")
    if cuda["torch_cuda_version"]:
        print(f"  torch CUDA version: {cuda['torch_cuda_version']}")
    if cuda["torch_gpu_names"]:
        print(f"  torch GPUs: {', '.join(cuda['torch_gpu_names'])}")
    print(f"  onnxruntime installed: {cuda['onnxruntime_installed']}")
    print(f"  onnxruntime providers: {', '.join(cuda['onnxruntime_providers']) or 'none'}")
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
    args = parser.parse_args()
    raise SystemExit(run_doctor(Path(args.config), strict=args.strict, json_output=args.json))


if __name__ == "__main__":
    main()
