from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .dependency_manager import acceleration_install_decision, cuda_diagnostics, dependency_status


def run_doctor(config_path: Path, strict: bool = False) -> int:
    config = load_config(config_path)
    folders = config["folders"]
    for folder in folders.values():
        Path(folder).mkdir(parents=True, exist_ok=True)
    status = dependency_status(config)
    print("Easy ASR Bench Doctor")
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
    cuda = cuda_diagnostics()
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
    for key, folder in folders.items():
        print(f"  {key}: {folder}")
    return 0 if (status["core"]["available"] or not strict) else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_doctor(Path(args.config), strict=args.strict))


if __name__ == "__main__":
    main()
