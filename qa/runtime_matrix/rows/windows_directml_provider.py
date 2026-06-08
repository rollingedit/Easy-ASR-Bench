from __future__ import annotations

from pathlib import Path

from app import dependency_manager as dm
from app.dependency_manager import cuda_diagnostics, install_group_for_config, recovery_command_for_config
from qa.runtime_matrix.common import package_versions, write_row


def _directml_conflict_repair_contract(row_id: str, evidence_dir: Path) -> dict:
    state = {"plain_onnxruntime": True, "provider_visible": False}
    commands: list[list[str]] = []
    original = {
        "nvidia_gpu_detected": dm.nvidia_gpu_detected,
        "intel_gpu_or_npu_detected": dm.intel_gpu_or_npu_detected,
        "windows_gpu_detected": dm.windows_gpu_detected,
        "amd_gpu_detected": dm.amd_gpu_detected,
        "_missing_import_modules": dm._missing_import_modules,
        "requirement_version_issues": dm.requirement_version_issues,
        "distribution_installed": dm.distribution_installed,
        "onnxruntime_available_providers": dm.onnxruntime_available_providers,
        "_run_dependency_command": dm._run_dependency_command,
    }

    def fake_distribution_installed(package: str) -> bool:
        if package == "onnxruntime":
            return state["plain_onnxruntime"]
        return False

    def fake_providers() -> tuple[list[str], str]:
        if state["provider_visible"]:
            return ["DmlExecutionProvider", "CPUExecutionProvider"], ""
        return ["AzureExecutionProvider", "CPUExecutionProvider"], "DmlExecutionProvider not listed"

    def fake_run_dependency_command(command: list[str], env, log_handle) -> None:
        commands.append(list(command))
        if command[-3:] == ["uninstall", "-y", "onnxruntime"]:
            state["plain_onnxruntime"] = False
        if command[-1] == "onnxruntime-directml==1.24.4" and "--force-reinstall" in command:
            state["provider_visible"] = True
        if log_handle is not None:
            log_handle.write("> " + " ".join(command) + "\n")

    config = {"runtime": {"provider": "auto"}, "dependency_install": {"allow_accelerator_install": True}}
    try:
        dm.nvidia_gpu_detected = lambda: False
        dm.intel_gpu_or_npu_detected = lambda: False
        dm.windows_gpu_detected = lambda: True
        dm.amd_gpu_detected = lambda: True
        dm._missing_import_modules = lambda metadata: []
        dm.requirement_version_issues = lambda requirement_files, ignored_packages=None: []
        dm.distribution_installed = fake_distribution_installed
        dm.onnxruntime_available_providers = fake_providers
        dm._run_dependency_command = fake_run_dependency_command
        missing_before = dm.missing_modules_for_config("onnx", config)
        repair_log = evidence_dir / "directml_conflict_repair.log"
        repair_result = dm.install_group_for_config("onnx", Path.cwd(), config, log_path=repair_log)
        missing_after = dm.missing_modules_for_config("onnx", config)
    finally:
        dm.nvidia_gpu_detected = original["nvidia_gpu_detected"]
        dm.intel_gpu_or_npu_detected = original["intel_gpu_or_npu_detected"]
        dm.windows_gpu_detected = original["windows_gpu_detected"]
        dm.amd_gpu_detected = original["amd_gpu_detected"]
        dm._missing_import_modules = original["_missing_import_modules"]
        dm.requirement_version_issues = original["requirement_version_issues"]
        dm.distribution_installed = original["distribution_installed"]
        dm.onnxruntime_available_providers = original["onnxruntime_available_providers"]
        dm._run_dependency_command = original["_run_dependency_command"]

    failures = []
    if "onnxruntime conflicts with directml package" not in missing_before:
        failures.append("DirectML conflict marker missing before repair")
    if "onnxruntime DirectML provider" not in missing_before:
        failures.append("DirectML missing-provider marker missing before repair")
    if missing_after:
        failures.append("DirectML repair contract did not clear missing markers: " + ", ".join(missing_after))
    if repair_result.get("accelerator") != "directml":
        failures.append("repair result did not select DirectML accelerator")
    if repair_result.get("provider_compatibility_repair") != "onnxruntime-directml==1.24.4":
        failures.append("repair result did not record the provider compatibility package")
    if not any(command[-3:] == ["uninstall", "-y", "onnxruntime"] for command in commands):
        failures.append("repair did not remove conflicting plain onnxruntime")
    if not any(command[-2:] == ["-r", str(Path.cwd() / "requirements" / "onnx_directml.txt")] for command in commands):
        failures.append("repair did not install the DirectML requirement file")
    if not any(command[-4:] == ["--upgrade", "--force-reinstall", "--no-deps", "onnxruntime-directml==1.24.4"] for command in commands):
        failures.append("repair did not force-probe the compatible DirectML provider wheel")
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "DirectML ONNX repair contract removes conflicting plain onnxruntime, installs the DirectML package, and records provider compatibility repair evidence."
            if not failures
            else "DirectML ONNX repair contract failed."
        ),
        details={
            "simulated_initial_state": {
                "windows_gpu_detected": True,
                "amd_gpu_detected": True,
                "plain_onnxruntime_installed": True,
                "providers": ["AzureExecutionProvider", "CPUExecutionProvider"],
            },
            "missing_before": missing_before,
            "missing_after": missing_after,
            "repair_result": repair_result,
            "commands": commands,
            "failures": failures,
        },
        artifacts=[repair_log],
    )


def run(row_id: str, evidence_dir: Path, install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id == "directml_provider_conflict_repair":
        return _directml_conflict_repair_contract(row_id, evidence_dir)
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
