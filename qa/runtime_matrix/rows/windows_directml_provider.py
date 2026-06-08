from __future__ import annotations

from pathlib import Path

from app.dependency_manager import cuda_diagnostics, install_group_for_config, recovery_command_for_config
from qa.runtime_matrix.common import package_versions, write_row


def run(row_id: str, evidence_dir: Path, install_deps: bool, _allow_downloads: bool) -> dict:
    diagnostics = cuda_diagnostics()
    repair_log = evidence_dir / "directml_repair.log"
    repair_result = None
    if (
        install_deps
        and diagnostics.get("windows_gpu_detected", False)
        and "DmlExecutionProvider" not in diagnostics.get("onnxruntime_providers", [])
    ):
        try:
            repair_result = install_group_for_config(
                "onnx",
                Path.cwd(),
                {"runtime": {"provider": "directml"}, "dependency_install": {"allow_accelerator_install": True}},
                log_path=repair_log,
            )
        except Exception as exc:
            repair_result = {"error_type": type(exc).__name__, "error": str(exc)}
        diagnostics = cuda_diagnostics()
    providers = diagnostics.get("onnxruntime_providers", [])
    details = {
        "provider": "DmlExecutionProvider",
        "cuda_provider_checks": diagnostics,
        "dependency_versions": package_versions(["onnxruntime", "onnxruntime-directml"]),
        "repair_command": recovery_command_for_config(
            "onnx",
            {"runtime": {"provider": "directml"}, "dependency_install": {"allow_accelerator_install": True}},
        ),
    }
    if repair_result is not None:
        details["repair_result"] = repair_result
    if "DmlExecutionProvider" in providers:
        return write_row(
            row_id,
            "pass",
            evidence_dir,
            summary="ONNX Runtime DirectML provider is visible for iGPU/Windows GPU validation.",
            details=details,
            artifacts=[repair_log],
        )
    if not diagnostics.get("windows_gpu_detected", False):
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="No usable Windows GPU was detected for DirectML validation.",
            block_reason="No non-basic Windows GPU detected.",
            external_requirement="Windows DirectML-capable GPU",
            details=details,
        )
    return write_row(
        row_id,
        "blocked",
        evidence_dir,
        summary="A Windows GPU was detected, but ONNX Runtime DirectML provider is not currently available.",
        block_reason="DmlExecutionProvider missing from onnxruntime providers",
        external_requirement=details["repair_command"],
        details=details,
        artifacts=[repair_log],
    )
