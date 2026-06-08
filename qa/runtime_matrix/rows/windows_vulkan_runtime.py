from __future__ import annotations

from pathlib import Path

from app.dependency_manager import cuda_diagnostics, recovery_command_for_config
from qa.runtime_matrix.common import package_versions, write_row


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    diagnostics = cuda_diagnostics()
    details = {
        "cuda_provider_checks": diagnostics,
        "dependency_versions": package_versions(["llama-cpp-python"]),
        "repair_command": recovery_command_for_config(
            "llama_cpp",
            {"runtime": {"provider": "vulkan"}, "dependency_install": {"allow_accelerator_install": True}},
        ),
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
