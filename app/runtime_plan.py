from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareInfo:
    nvidia: bool = False
    amd: bool = False
    intel_gpu: bool = False
    windows_gpu: bool = False
    vulkan_runtime: bool = False
    vulkan_sdk: bool = False
    torch_cuda_available: bool = False
    onnx_providers: tuple[str, ...] = ()
    llama_cpp_gpu_offload: bool = False
    ctranslate2_cuda_available: bool = False


@dataclass(frozen=True)
class ResolvedRuntimePlan:
    model_family: str
    requested_provider: str
    actual_provider: str
    device: str
    compute_type: str | None
    backend_verified: bool
    fallback_allowed: bool
    reason: str
    fallback_reason: str | None = None


def hardware_from_dependency_manager() -> HardwareInfo:
    from . import dependency_manager as deps

    diagnostics = deps.cuda_diagnostics()
    return HardwareInfo(
        nvidia=bool(diagnostics.get("nvidia_gpu_detected", False)),
        amd=bool(diagnostics.get("amd_gpu_detected", False)),
        intel_gpu=bool(diagnostics.get("intel_gpu_or_npu_detected", False)),
        windows_gpu=bool(diagnostics.get("windows_gpu_detected", False)),
        vulkan_runtime=bool(diagnostics.get("vulkan_detected", False)),
        vulkan_sdk=bool(diagnostics.get("vulkan_sdk_detected", False)),
        torch_cuda_available=bool(diagnostics.get("torch_cuda_available", False)),
        onnx_providers=tuple(diagnostics.get("onnxruntime_providers", []) or ()),
        llama_cpp_gpu_offload=deps.llama_cpp_gpu_capable(),
        ctranslate2_cuda_available=deps.ctranslate2_cuda_available(),
    )


def resolve_runtime_plan(model_family: str, runtime_config: dict, hardware: HardwareInfo | None = None) -> ResolvedRuntimePlan:
    hardware = hardware or hardware_from_dependency_manager()
    runtime = runtime_config.get("runtime", runtime_config)
    requested = str(runtime.get("provider", "auto")).lower()
    prefer_gpu = bool(runtime.get("prefer_gpu", True))
    fallback_allowed = bool(runtime.get("fallback_to_cpu", True))

    if model_family == "faster_whisper":
        wants_cuda = requested == "cuda" or requested == "auto" and prefer_gpu and hardware.nvidia
        if wants_cuda and hardware.ctranslate2_cuda_available:
            return ResolvedRuntimePlan(model_family, requested, "cuda", "cuda", None, True, fallback_allowed, "CTranslate2 CUDA backend is available.")
        if wants_cuda:
            return ResolvedRuntimePlan(
                model_family,
                requested,
                "cpu",
                "cpu",
                None,
                False,
                fallback_allowed,
                "CUDA was requested/preferred, but CTranslate2 CUDA backend was not verified.",
                "Using CPU for faster-whisper until CUDA/CTranslate2 runtime is verified.",
            )
        return ResolvedRuntimePlan(model_family, requested, "cpu", "cpu", None, True, fallback_allowed, "CPU runtime selected.")

    if model_family == "transformers_asr":
        wants_cuda = requested == "cuda" or requested == "auto" and prefer_gpu and hardware.nvidia
        if wants_cuda and hardware.torch_cuda_available:
            return ResolvedRuntimePlan(model_family, requested, "cuda", "cuda", None, True, fallback_allowed, "Torch CUDA is available.")
        if wants_cuda:
            return ResolvedRuntimePlan(
                model_family,
                requested,
                "cpu",
                "cpu",
                None,
                False,
                fallback_allowed,
                "CUDA was requested/preferred, but Torch CUDA was not verified.",
                "Using CPU for Transformers ASR until a CUDA-enabled Torch install is verified.",
            )
        return ResolvedRuntimePlan(model_family, requested, "cpu", "cpu", None, True, fallback_allowed, "CPU runtime selected.")

    if model_family == "llama_cpp":
        if requested == "cuda" or requested == "auto" and prefer_gpu and hardware.nvidia:
            if hardware.llama_cpp_gpu_offload:
                return ResolvedRuntimePlan(model_family, requested, "cuda", "cuda", None, True, fallback_allowed, "llama-cpp-python GPU offload is available.")
            return ResolvedRuntimePlan(
                model_family,
                requested,
                "cpu",
                "cpu",
                None,
                False,
                fallback_allowed,
                "CUDA/prefer_gpu was requested, but llama-cpp-python GPU offload was not verified.",
                "Using n_gpu_layers=0 to avoid a broken GPU offload path.",
            )
        if requested == "vulkan" or requested == "auto" and prefer_gpu and hardware.vulkan_runtime:
            if hardware.vulkan_sdk and hardware.llama_cpp_gpu_offload:
                return ResolvedRuntimePlan(model_family, requested, "vulkan", "vulkan", None, True, fallback_allowed, "llama.cpp Vulkan backend is verified.")
            return ResolvedRuntimePlan(
                model_family,
                requested,
                "cpu",
                "cpu",
                None,
                False,
                fallback_allowed,
                "Vulkan runtime is visible, but Vulkan SDK/backend verification is missing.",
                "Using CPU until a Vulkan-capable llama.cpp backend is installed.",
            )
        return ResolvedRuntimePlan(model_family, requested, "cpu", "cpu", None, True, fallback_allowed, "CPU runtime selected.")

    return ResolvedRuntimePlan(model_family, requested, "cpu", "cpu", None, True, fallback_allowed, "No specialized runtime plan exists.")
