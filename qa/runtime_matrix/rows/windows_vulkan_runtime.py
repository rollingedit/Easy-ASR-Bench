from __future__ import annotations

from pathlib import Path

from app.dependency_manager import cuda_diagnostics, install_group_for_config, llama_cpp_gpu_capable, missing_modules_for_config, recovery_command_for_config
from qa.runtime_matrix.common import ROOT, package_versions, write_row
from qa.runtime_matrix.rows.gguf_reference_llm_smollm135 import SMOLLM_PATH


VULKAN_CONFIG = {
    "runtime": {"provider": "vulkan", "prefer_gpu": True, "fallback_to_cpu": True},
    "dependency_install": {"allow_accelerator_install": True},
}


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id == "llama_cpp_vulkan_smollm_smoke":
        return _llama_cpp_vulkan_smollm_smoke(row_id, evidence_dir, _install_deps)
    diagnostics = cuda_diagnostics()
    details = {
        "cuda_provider_checks": diagnostics,
        "dependency_versions": package_versions(["llama-cpp-python"]),
        "repair_command": recovery_command_for_config("llama_cpp", VULKAN_CONFIG),
    }
    if diagnostics.get("vulkan_detected", False):
        summary = "Vulkan runtime is visible for llama.cpp Vulkan validation."
        if not diagnostics.get("vulkan_sdk_detected", False):
            summary += " Vulkan SDK/build tooling is not detected, so source builds remain blocked unless the prebuilt wheel works."
        return write_row(row_id, "pass", evidence_dir, summary=summary, details=details)
    return write_row(
        row_id,
        "blocked",
        evidence_dir,
        summary="Vulkan runtime is not currently visible on this machine.",
        block_reason="vulkaninfo/runtime probe did not detect Vulkan",
        external_requirement="Vulkan runtime or a CPU fallback for llama-cpp-python",
        details=details,
    )


def _llama_cpp_vulkan_smollm_smoke(row_id: str, evidence_dir: Path, install_deps: bool) -> dict:
    diagnostics = cuda_diagnostics()
    log_path = evidence_dir / "llama_cpp_vulkan_repair.log"
    details = {
        "cuda_provider_checks": diagnostics,
        "dependency_versions": package_versions(["llama-cpp-python"]),
        "repair_command": recovery_command_for_config("llama_cpp", VULKAN_CONFIG),
        "install_deps_allowed": install_deps,
        "smollm_path": str(SMOLLM_PATH),
        "llama_cpp_gpu_offload_before": llama_cpp_gpu_capable(),
        "missing_before": missing_modules_for_config("llama_cpp", VULKAN_CONFIG),
    }
    if not diagnostics.get("vulkan_detected", False):
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="llama.cpp Vulkan SmolLM offload smoke requires a visible Vulkan runtime.",
            block_reason="vulkaninfo/runtime probe did not detect Vulkan",
            external_requirement="Windows machine with Vulkan runtime plus a Vulkan-capable llama-cpp-python wheel",
            details=details,
        )
    if details["missing_before"] and not install_deps:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="llama.cpp Vulkan SmolLM smoke needs dependency repair before offload can be verified.",
            block_reason="missing: " + ", ".join(details["missing_before"]),
            external_requirement="rerun with --install-deps to let the product dependency repair path try the Vulkan llama-cpp-python wheel",
            details=details,
        )
    if details["missing_before"] and install_deps:
        try:
            details["repair_result"] = install_group_for_config("llama_cpp", ROOT, VULKAN_CONFIG, log_path=log_path)
        except Exception as exc:
            details["repair_error"] = {"type": type(exc).__name__, "message": str(exc), "log_path": str(log_path)}
            return write_row(
                row_id,
                "blocked",
                evidence_dir,
                summary="llama.cpp Vulkan dependency repair failed before SmolLM offload could be verified.",
                block_reason=f"llama_cpp Vulkan repair failed: {type(exc).__name__}: {exc}",
                external_requirement="working network/package index or a Windows Vulkan machine where the prebuilt llama-cpp-python Vulkan wheel installs",
                details=details,
                artifacts=[log_path],
            )
    details["dependency_versions_after"] = package_versions(["llama-cpp-python"])
    details["missing_after"] = missing_modules_for_config("llama_cpp", VULKAN_CONFIG)
    details["llama_cpp_gpu_offload_after"] = llama_cpp_gpu_capable()
    if details["missing_after"] or not details["llama_cpp_gpu_offload_after"]:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="llama.cpp imports but Vulkan/GPU offload is not verified on this machine.",
            block_reason="missing: " + ", ".join(details["missing_after"] or ["llama-cpp-python GPU offload build"]),
            external_requirement="Vulkan-capable llama-cpp-python wheel or explicit source-build environment with Vulkan SDK/build tooling",
            details=details,
            artifacts=[log_path],
        )
    if not SMOLLM_PATH.exists():
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="llama.cpp Vulkan offload is available, but the SmolLM fixture is not staged.",
            block_reason="missing SmolLM 135M GGUF fixture",
            external_requirement="run gguf_reference_llm or stage Temp/real_tiny_llm_smoke/Models/SmolLM-135M-GGUF/SmolLM-135M.Q4_K_M.gguf",
            details=details,
            artifacts=[log_path],
        )
    try:
        from app.dependency_manager import prepare_llama_cpp_dll_search_path

        prepare_llama_cpp_dll_search_path()
        from llama_cpp import Llama

        llm = Llama(model_path=str(SMOLLM_PATH), n_ctx=256, n_gpu_layers=-1, verbose=False)
        output = llm("Say one short word.", max_tokens=8, temperature=0)
        details["generated_text"] = output["choices"][0]["text"].strip()
    except Exception as exc:
        details["smoke_error"] = {"type": type(exc).__name__, "message": str(exc)}
        return write_row(row_id, "fail", evidence_dir, summary="llama.cpp Vulkan/offload SmolLM smoke failed after offload capability was reported.", details=details, artifacts=[SMOLLM_PATH, log_path])
    if not details["generated_text"]:
        return write_row(row_id, "fail", evidence_dir, summary="llama.cpp Vulkan/offload SmolLM smoke returned empty text.", details=details, artifacts=[SMOLLM_PATH, log_path])
    return write_row(row_id, "pass", evidence_dir, summary="llama.cpp Vulkan/offload SmolLM smoke passed.", details=details, artifacts=[SMOLLM_PATH, log_path])
