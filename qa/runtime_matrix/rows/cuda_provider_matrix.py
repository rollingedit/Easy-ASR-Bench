from __future__ import annotations

import importlib.metadata
from pathlib import Path

import sys

from app.dependency_manager import CUDA_INSTALL_OVERRIDES, cuda_diagnostics, llama_cpp_gpu_capable, recovery_command_for_config
from app.runtime_plan import hardware_from_dependency_manager, resolve_runtime_plan
from qa.runtime_matrix.common import package_versions, write_row


def _repair_commands() -> dict[str, str]:
    cuda_config = {"runtime": {"provider": "cuda"}, "dependency_install": {"allow_accelerator_install": True}}
    return {
        "torch_transformers": recovery_command_for_config("transformers_cpu", cuda_config),
        "onnx_cuda": recovery_command_for_config("onnx", cuda_config),
        "faster_whisper_cuda": recovery_command_for_config("faster_whisper", cuda_config),
        "llama_cpp_cuda": recovery_command_for_config("llama_cpp", cuda_config),
    }


def _explicit_cuda_requirement_commands() -> dict[str, list[str]]:
    return {
        group: [f'"{sys.executable}" -m pip install -r {requirement}' for requirement in override["requirement_files"]]
        for group, override in CUDA_INSTALL_OVERRIDES.items()
    }


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_cuda_combo(row_id: str, evidence_dir: Path) -> dict:
    diagnostics = cuda_diagnostics()
    details = {
        "cuda_provider_checks": diagnostics,
        "dependency_versions": package_versions(
            [
                "torch",
                "onnxruntime",
                "onnxruntime-gpu",
                "ctranslate2",
                "faster-whisper",
                "llama-cpp-python",
            ]
        ),
        "llama_cpp_gpu_offload": llama_cpp_gpu_capable(),
        "repair_commands": _repair_commands(),
        "explicit_cuda_requirement_commands": _explicit_cuda_requirement_commands(),
        "installed_cuda_runtime_packages": {
            name: _installed_version(name)
            for name in [
                "nvidia-cublas-cu12",
                "nvidia-cudnn-cu12",
                "nvidia-cuda-runtime-cu12",
            ]
            if _installed_version(name)
        },
    }
    missing: list[str] = []
    if not diagnostics.get("nvidia_gpu_detected", False):
        missing.append("NVIDIA CUDA-capable GPU")
    if not diagnostics.get("torch_cuda_available", False):
        missing.append("Torch CUDA")
    if not diagnostics.get("onnx_cuda_available", False):
        missing.append("ONNX Runtime CUDAExecutionProvider")
    if not diagnostics.get("ctranslate2_cuda_available", False):
        missing.append("CTranslate2 CUDA backend")
    if not details["llama_cpp_gpu_offload"]:
        missing.append("llama-cpp-python GPU offload backend")
    if missing:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="CUDA combined provider row has a real detector, but this machine does not currently satisfy every CUDA backend requirement.",
            block_reason="missing: " + ", ".join(missing),
            external_requirement="NVIDIA CUDA machine with verified Torch CUDA, ONNX Runtime CUDA, CTranslate2 CUDA, and llama-cpp-python GPU offload",
            details=details,
        )
    return write_row(
        row_id,
        "pass",
        evidence_dir,
        summary="NVIDIA CUDA provider prerequisites are visible for Torch, ONNX Runtime, CTranslate2, and llama-cpp-python.",
        details=details,
    )


def _faster_whisper_cuda_fallback(row_id: str, evidence_dir: Path) -> dict:
    hardware = hardware_from_dependency_manager()
    plan = resolve_runtime_plan(
        "faster_whisper",
        {"runtime": {"provider": "cuda", "prefer_gpu": True, "fallback_to_cpu": True}},
        hardware,
    )
    details = {
        "plan": {
            "model_family": plan.model_family,
            "requested_provider": plan.requested_provider,
            "actual_provider": plan.actual_provider,
            "device": plan.device,
            "backend_verified": plan.backend_verified,
            "fallback_allowed": plan.fallback_allowed,
            "reason": plan.reason,
            "fallback_reason": plan.fallback_reason,
        },
        "hardware": {
            "nvidia": hardware.nvidia,
            "ctranslate2_cuda_available": hardware.ctranslate2_cuda_available,
            "torch_cuda_available": hardware.torch_cuda_available,
            "onnx_providers": list(hardware.onnx_providers),
        },
        "cuda_provider_checks": cuda_diagnostics(),
        "dependency_versions": package_versions(["ctranslate2", "faster-whisper"]),
        "repair_command": _repair_commands()["faster_whisper_cuda"],
    }
    if plan.actual_provider == "cpu" and plan.fallback_reason and plan.fallback_allowed and not plan.backend_verified:
        return write_row(
            row_id,
            "pass",
            evidence_dir,
            summary="faster-whisper CUDA request resolves to explicit CPU fallback when CTranslate2 CUDA is not verified.",
            details=details,
        )
    if plan.actual_provider == "cuda" and plan.backend_verified:
        return write_row(
            row_id,
            "pass",
            evidence_dir,
            summary="faster-whisper CUDA request resolves to verified CUDA on this machine.",
            details=details,
        )
    return write_row(
        row_id,
        "fail",
        evidence_dir,
        summary="faster-whisper CUDA fallback plan was not explicit enough.",
        details=details,
    )


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id == "nvidia_cuda_torch_onnx_faster_whisper_llama":
        return _nvidia_cuda_combo(row_id, evidence_dir)
    if row_id == "faster_whisper_cuda_unavailable_cpu_fallback":
        return _faster_whisper_cuda_fallback(row_id, evidence_dir)
    return write_row(row_id, "fail", evidence_dir, summary=f"Unsupported CUDA provider row: {row_id}")
