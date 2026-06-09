from __future__ import annotations

import importlib.metadata
import json
import subprocess
from pathlib import Path

import sys

from app.dependency_manager import CUDA_INSTALL_OVERRIDES, acceleration_install_decision, cuda_diagnostics, llama_cpp_gpu_capable, recovery_command_for_config
from app.runtime_plan import hardware_from_dependency_manager, resolve_runtime_plan
from qa.runtime_matrix.common import package_versions, write_row
from qa.runtime_matrix.rows.gguf_reference_llm_smollm135 import SMOLLM_PATH
from qa.runtime_matrix.rows.generic_onnx_ctc_tiny import _write_tiny_ctc_fixture


def _repair_commands() -> dict[str, str]:
    cuda_config = {"runtime": {"provider": "cuda"}, "dependency_install": {"allow_accelerator_install": True}}
    return {
        "torch_transformers": recovery_command_for_config("transformers_cpu", cuda_config),
        "openai_whisper": recovery_command_for_config("openai_whisper", cuda_config),
        "onnx_cuda": recovery_command_for_config("onnx", cuda_config),
        "faster_whisper_cuda": recovery_command_for_config("faster_whisper", cuda_config),
        "llama_cpp_cuda": recovery_command_for_config("llama_cpp", cuda_config),
    }


def _explicit_cuda_requirement_commands() -> dict[str, list[str]]:
    commands = {
        group: [f'"{sys.executable}" -m pip install -r {requirement}' for requirement in override["requirement_files"]]
        for group, override in CUDA_INSTALL_OVERRIDES.items()
    }
    decision = acceleration_install_decision(
        {
            "runtime": {"provider": "cuda", "prefer_gpu": True},
            "dependency_install": {"allow_cuda_install": True, "allow_accelerator_install": True},
        },
        "llama_cpp",
    )
    repair_commands = decision.get("repair_commands") or []
    if repair_commands:
        if isinstance(repair_commands, dict):
            commands["llama_cpp"] = [str(command) for command in repair_commands.values()]
        else:
            commands["llama_cpp"] = [str(command) for command in repair_commands]
    return commands


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _last_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    last: dict = {}
    last_schema: dict = {}
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            last = value
            if "schema" in value:
                last_schema = value
    return last_schema or last


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


def _cuda_block(row_id: str, evidence_dir: Path, *, missing: list[str], details: dict, external_requirement: str) -> dict:
    return write_row(
        row_id,
        "blocked",
        evidence_dir,
        summary="CUDA row has real smoke logic, but this machine is missing required CUDA capability.",
        block_reason="missing: " + ", ".join(missing),
        external_requirement=external_requirement,
        details=details,
    )


def _torch_cuda_smoke(row_id: str, evidence_dir: Path) -> dict:
    diagnostics = cuda_diagnostics()
    details = {
        "cuda_provider_checks": diagnostics,
        "dependency_versions": package_versions(["torch"]),
        "repair_command": _repair_commands()["torch_transformers"],
        "explicit_cuda_requirement_commands": _explicit_cuda_requirement_commands().get("transformers_cpu", []),
    }
    missing: list[str] = []
    if not diagnostics.get("nvidia_gpu_detected", False):
        missing.append("NVIDIA CUDA-capable GPU")
    if not diagnostics.get("torch_cuda_available", False):
        missing.append("Torch CUDA")
    if missing:
        return _cuda_block(
            row_id,
            evidence_dir,
            missing=missing,
            details=details,
            external_requirement="NVIDIA CUDA machine with a CUDA-enabled Torch wheel and driver-compatible runtime",
        )
    try:
        import torch

        tensor = torch.ones((2, 2), device="cuda")
        torch.cuda.synchronize()
        details["tensor_device"] = str(tensor.device)
        details["tensor_sum"] = float(tensor.sum().item())
        details["cuda_device_name"] = torch.cuda.get_device_name(0)
        details["torch_cuda_version"] = torch.version.cuda
    except Exception as exc:
        details["smoke_error"] = {"type": type(exc).__name__, "message": str(exc)}
        return write_row(row_id, "fail", evidence_dir, summary="Torch CUDA was reported available but tiny tensor allocation failed.", details=details)
    return write_row(row_id, "pass", evidence_dir, summary="Torch CUDA tiny tensor allocation passed.", details=details)


def _onnx_cuda_smoke(row_id: str, evidence_dir: Path) -> dict:
    diagnostics = cuda_diagnostics()
    details = {
        "cuda_provider_checks": diagnostics,
        "dependency_versions": package_versions(["onnx", "onnxruntime", "onnxruntime-gpu"]),
        "repair_command": _repair_commands()["onnx_cuda"],
        "explicit_cuda_requirement_commands": _explicit_cuda_requirement_commands().get("onnx", []),
    }
    missing: list[str] = []
    if not diagnostics.get("nvidia_gpu_detected", False):
        missing.append("NVIDIA CUDA-capable GPU")
    if not diagnostics.get("onnx_cuda_available", False):
        missing.append("ONNX Runtime CUDAExecutionProvider")
    if missing:
        return _cuda_block(
            row_id,
            evidence_dir,
            missing=missing,
            details=details,
            external_requirement="NVIDIA CUDA machine with onnxruntime-gpu exposing CUDAExecutionProvider",
        )
    try:
        import numpy as np
        import onnxruntime as ort

        model_dir = evidence_dir / "tiny_onnx_cuda"
        artifacts = _write_tiny_ctc_fixture(model_dir)
        session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        output = session.run(None, {"input_values": np.zeros((1, 1600), dtype=np.float32)})
        details["active_providers"] = session.get_providers()
        details["output_shape"] = list(output[0].shape)
    except Exception as exc:
        details["smoke_error"] = {"type": type(exc).__name__, "message": str(exc)}
        return write_row(row_id, "fail", evidence_dir, summary="ONNX Runtime CUDA provider was visible but tiny session execution failed.", details=details)
    if "CUDAExecutionProvider" not in details["active_providers"]:
        return write_row(row_id, "fail", evidence_dir, summary="ONNX Runtime CUDA smoke fell back without using CUDAExecutionProvider.", details=details)
    return write_row(row_id, "pass", evidence_dir, summary="ONNX Runtime CUDA tiny session passed.", details=details, artifacts=artifacts)


def _faster_whisper_cuda_smoke(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    diagnostics = cuda_diagnostics()
    details = {
        "cuda_provider_checks": diagnostics,
        "dependency_versions": package_versions(["ctranslate2", "faster-whisper"]),
        "repair_command": _repair_commands()["faster_whisper_cuda"],
        "explicit_cuda_requirement_commands": _explicit_cuda_requirement_commands().get("faster_whisper", []),
    }
    missing: list[str] = []
    if not diagnostics.get("nvidia_gpu_detected", False):
        missing.append("NVIDIA CUDA-capable GPU")
    if not diagnostics.get("ctranslate2_cuda_available", False):
        missing.append("CTranslate2 CUDA backend")
    if missing:
        return _cuda_block(
            row_id,
            evidence_dir,
            missing=missing,
            details=details,
            external_requirement="NVIDIA CUDA machine with faster-whisper and a verified CTranslate2 CUDA backend",
        )
    if not allow_downloads:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="faster-whisper CUDA smoke needs the tiny model fixture and downloads are disabled.",
            block_reason="downloads disabled for Systran/faster-whisper-tiny.en fixture",
            external_requirement="rerun with --allow-downloads on a CUDA machine",
            details=details,
        )
    command = [
        sys.executable,
        "qa/run_real_tiny_model_smoke.py",
        "--provider",
        "cuda",
        "--workdir",
        str(evidence_dir / "real_tiny_faster_whisper_cuda_smoke"),
    ]
    if install_deps:
        command.append("--install-deps")
    completed = subprocess.run(command, text=True, capture_output=True, timeout=900)
    details["command"] = {
        "command": command,
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        return write_row(row_id, "fail", evidence_dir, summary="faster-whisper CUDA app-pipeline smoke failed.", details=details)
    payload = _last_json_object(completed.stdout)
    details["smoke_payload"] = payload
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    if metrics.get("provider_summary", {}).get("actual_provider") not in {"cuda", "auto"} and metrics.get("device") != "cuda":
        return write_row(row_id, "fail", evidence_dir, summary="faster-whisper CUDA smoke completed without CUDA provider evidence.", details=details)
    return write_row(row_id, "pass", evidence_dir, summary="faster-whisper/CTranslate2 CUDA app-pipeline smoke passed.", details=details)


def _llama_cpp_cuda_smoke(row_id: str, evidence_dir: Path) -> dict:
    diagnostics = cuda_diagnostics()
    gpu_capable = llama_cpp_gpu_capable()
    details = {
        "cuda_provider_checks": diagnostics,
        "llama_cpp_gpu_offload": gpu_capable,
        "dependency_versions": package_versions(["llama-cpp-python"]),
        "repair_command": _repair_commands()["llama_cpp_cuda"],
        "explicit_cuda_requirement_commands": _explicit_cuda_requirement_commands().get("llama_cpp", []),
        "smollm_path": str(SMOLLM_PATH),
    }
    missing: list[str] = []
    if not diagnostics.get("nvidia_gpu_detected", False):
        missing.append("NVIDIA CUDA-capable GPU")
    if not gpu_capable:
        missing.append("llama-cpp-python GPU offload backend")
    if not SMOLLM_PATH.exists():
        missing.append("SmolLM 135M GGUF fixture")
    if missing:
        return _cuda_block(
            row_id,
            evidence_dir,
            missing=missing,
            details=details,
            external_requirement="NVIDIA CUDA machine with CUDA-capable llama-cpp-python and cached SmolLM 135M GGUF",
        )
    try:
        from app.dependency_manager import prepare_llama_cpp_dll_search_path

        prepare_llama_cpp_dll_search_path()
        from llama_cpp import Llama

        llm = Llama(model_path=str(SMOLLM_PATH), n_ctx=256, n_gpu_layers=-1, verbose=False)
        output = llm("Say one short word.", max_tokens=8, temperature=0)
        text = output["choices"][0]["text"].strip()
        details["generated_text"] = text
    except Exception as exc:
        details["smoke_error"] = {"type": type(exc).__name__, "message": str(exc)}
        return write_row(row_id, "fail", evidence_dir, summary="llama.cpp CUDA/offload SmolLM smoke failed.", details=details, artifacts=[SMOLLM_PATH])
    if not details.get("generated_text"):
        return write_row(row_id, "fail", evidence_dir, summary="llama.cpp CUDA/offload SmolLM smoke returned empty text.", details=details, artifacts=[SMOLLM_PATH])
    return write_row(row_id, "pass", evidence_dir, summary="llama.cpp CUDA/offload SmolLM smoke passed.", details=details, artifacts=[SMOLLM_PATH])


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


def _transformers_cuda_fallback(row_id: str, evidence_dir: Path) -> dict:
    hardware = hardware_from_dependency_manager()
    plan = resolve_runtime_plan(
        "transformers_asr",
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
            "torch_cuda_available": hardware.torch_cuda_available,
            "onnx_providers": list(hardware.onnx_providers),
        },
        "cuda_provider_checks": cuda_diagnostics(),
        "dependency_versions": package_versions(["torch", "transformers", "safetensors", "sentencepiece", "protobuf", "torchaudio"]),
        "repair_command": _repair_commands()["torch_transformers"],
        "explicit_cuda_requirement_commands": _explicit_cuda_requirement_commands().get("transformers_cpu", []),
    }
    if plan.actual_provider == "cpu" and plan.fallback_reason and plan.fallback_allowed and not plan.backend_verified:
        return write_row(
            row_id,
            "pass",
            evidence_dir,
            summary="Transformers ASR CUDA request resolves to explicit CPU fallback when Torch CUDA is not verified.",
            details=details,
        )
    if plan.actual_provider == "cuda" and plan.backend_verified:
        return write_row(
            row_id,
            "pass",
            evidence_dir,
            summary="Transformers ASR CUDA request resolves to verified Torch CUDA on this machine.",
            details=details,
        )
    return write_row(
        row_id,
        "fail",
        evidence_dir,
        summary="Transformers ASR CUDA fallback plan was not explicit enough.",
        details=details,
    )


def _openai_whisper_cuda_fallback(row_id: str, evidence_dir: Path) -> dict:
    hardware = hardware_from_dependency_manager()
    plan = resolve_runtime_plan(
        "openai_whisper",
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
            "torch_cuda_available": hardware.torch_cuda_available,
        },
        "cuda_provider_checks": cuda_diagnostics(),
        "dependency_versions": package_versions(["torch", "openai-whisper"]),
        "repair_command": _repair_commands()["openai_whisper"],
        "explicit_cuda_requirement_commands": _explicit_cuda_requirement_commands().get("openai_whisper", []),
    }
    if plan.actual_provider == "cpu" and plan.fallback_reason and plan.fallback_allowed and not plan.backend_verified:
        return write_row(
            row_id,
            "pass",
            evidence_dir,
            summary="OpenAI Whisper .pt CUDA request resolves to explicit CPU fallback when Torch CUDA is not verified.",
            details=details,
        )
    if plan.actual_provider == "cuda" and plan.backend_verified:
        return write_row(
            row_id,
            "pass",
            evidence_dir,
            summary="OpenAI Whisper .pt CUDA request resolves to verified Torch CUDA on this machine.",
            details=details,
        )
    return write_row(
        row_id,
        "fail",
        evidence_dir,
        summary="OpenAI Whisper .pt CUDA fallback plan was not explicit enough.",
        details=details,
    )


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id == "nvidia_cuda_torch_onnx_faster_whisper_llama":
        return _nvidia_cuda_combo(row_id, evidence_dir)
    if row_id == "nvidia_cuda_hardware_detection":
        return _nvidia_cuda_combo(row_id, evidence_dir)
    if row_id == "torch_cuda_tensor_smoke":
        return _torch_cuda_smoke(row_id, evidence_dir)
    if row_id == "onnxruntime_cuda_tiny_session":
        return _onnx_cuda_smoke(row_id, evidence_dir)
    if row_id == "faster_whisper_ctranslate2_cuda_smoke":
        return _faster_whisper_cuda_smoke(row_id, evidence_dir, _install_deps, _allow_downloads)
    if row_id == "llama_cpp_cuda_smollm_smoke":
        return _llama_cpp_cuda_smoke(row_id, evidence_dir)
    if row_id == "faster_whisper_cuda_unavailable_cpu_fallback":
        return _faster_whisper_cuda_fallback(row_id, evidence_dir)
    if row_id == "transformers_cuda_unavailable_cpu_fallback":
        return _transformers_cuda_fallback(row_id, evidence_dir)
    if row_id == "openai_whisper_cuda_unavailable_cpu_fallback":
        return _openai_whisper_cuda_fallback(row_id, evidence_dir)
    return write_row(row_id, "fail", evidence_dir, summary=f"Unsupported CUDA provider row: {row_id}")
