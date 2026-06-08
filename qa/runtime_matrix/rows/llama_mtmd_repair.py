from __future__ import annotations

from pathlib import Path

from app import repair_plan
from app.dependency_manager import LLAMA_MTMD_REPAIR_COMMAND
from qa.runtime_matrix.common import write_row


MISSING_MTMD = ["llama-mtmd-cli or llama-cpp-python Qwen3ASRChatHandler"]


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id != "llama_mtmd_dependency_repair_contract":
        return write_row(row_id, "fail", evidence_dir, summary=f"Unsupported llama-mtmd repair row: {row_id}")
    return _llama_mtmd_dependency_repair_contract(row_id, evidence_dir)


def _llama_mtmd_dependency_repair_contract(row_id: str, evidence_dir: Path) -> dict:
    installed = {"value": False}
    commands: list[str] = []
    cli_path = evidence_dir / "Runtime" / "llama-mtmd-cli.exe"
    original_dependency_status = repair_plan.dependency_status
    original_acceleration_install_decision = repair_plan.acceleration_install_decision
    original_install_group_for_config = repair_plan.install_group_for_config
    original_missing_modules_for_config = repair_plan.missing_modules_for_config
    original_backend_probe_for_group = repair_plan.backend_probe_for_group

    def fake_dependency_status(config):
        return {
            "llama_mtmd": {
                "available": installed["value"],
                "missing": [] if installed["value"] else list(MISSING_MTMD),
                "install_kind": "native_tool",
                "requirement_file": "",
                "recovery_command": LLAMA_MTMD_REPAIR_COMMAND,
                "accelerator_recovery_command": "",
            }
        }

    def fake_install_group_for_config(group, project_root, config, log_path=None):
        if group != "llama_mtmd":
            raise RuntimeError(f"unexpected repair group: {group}")
        commands.append(LLAMA_MTMD_REPAIR_COMMAND)
        cli_path.parent.mkdir(parents=True, exist_ok=True)
        cli_path.write_text("fixture native runtime\n", encoding="utf-8", newline="\n")
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(LLAMA_MTMD_REPAIR_COMMAND + "\n", encoding="utf-8", newline="\n")
        installed["value"] = True
        return {
            "installed_group": group,
            "native_tool": "llama-mtmd-cli",
            "repair_command": LLAMA_MTMD_REPAIR_COMMAND,
            "post_install_status": {
                "available": True,
                "path": str(cli_path),
                "configured_path": "",
                "qwen3_asr_handler_available": False,
                "missing": [],
            },
        }

    def fake_missing_modules_for_config(group, config):
        if group == "llama_mtmd":
            return [] if installed["value"] else list(MISSING_MTMD)
        return []

    def fake_backend_probe_for_group(group, config):
        if group != "llama_mtmd":
            raise RuntimeError(f"unexpected probe group: {group}")
        return {
            "kind": "llama_mtmd_runtime_probe",
            "ok": installed["value"],
            "runtime_status": {
                "available": installed["value"],
                "path": str(cli_path) if installed["value"] else "",
                "configured_path": "",
                "qwen3_asr_handler_available": False,
                "repair_command": LLAMA_MTMD_REPAIR_COMMAND,
                "missing": [] if installed["value"] else list(MISSING_MTMD),
            },
        }

    try:
        repair_plan.dependency_status = fake_dependency_status
        repair_plan.acceleration_install_decision = lambda config, group: {"use_accelerator": False, "accelerator": None}
        repair_plan.install_group_for_config = fake_install_group_for_config
        repair_plan.missing_modules_for_config = fake_missing_modules_for_config
        repair_plan.backend_probe_for_group = fake_backend_probe_for_group
        plan = repair_plan.execute_repair_plan(
            {"runtime": {"provider": "cpu", "prefer_gpu": False}, "dependency_install": {"use_cached_runtime_resolutions": False}},
            project_root=evidence_dir,
        )
    finally:
        repair_plan.dependency_status = original_dependency_status
        repair_plan.acceleration_install_decision = original_acceleration_install_decision
        repair_plan.install_group_for_config = original_install_group_for_config
        repair_plan.missing_modules_for_config = original_missing_modules_for_config
        repair_plan.backend_probe_for_group = original_backend_probe_for_group

    record = plan["records"][0]
    backend_probe = record.get("after", {}).get("backend_probe", {})
    runtime_status = backend_probe.get("runtime_status", {}) if isinstance(backend_probe, dict) else {}
    runtime_resolution_path = record.get("after", {}).get("runtime_resolution_path", "")
    runtime_resolution = record.get("after", {}).get("runtime_resolution", {})
    failures = []
    if record.get("affected_dependency_group") != "llama_mtmd":
        failures.append("repair record did not target llama_mtmd")
    if record.get("before", {}).get("install_kind") != "native_tool":
        failures.append("llama_mtmd was not classified as a native_tool repair")
    if record.get("status") != "repaired":
        failures.append("llama_mtmd repair did not finish as repaired")
    if record.get("repair_action") != "install_missing":
        failures.append("repair action was not install_missing")
    if record.get("repair_command") != LLAMA_MTMD_REPAIR_COMMAND:
        failures.append("repair command did not use the llama-mtmd native repair command")
    if record.get("after", {}).get("missing") != []:
        failures.append("post-repair missing llama-mtmd runtime was not cleared")
    if backend_probe.get("kind") != "llama_mtmd_runtime_probe":
        failures.append("post-repair backend probe was not the llama-mtmd runtime probe")
    if not runtime_status.get("path"):
        failures.append("post-repair llama-mtmd probe did not record a CLI path")
    if not runtime_resolution_path:
        failures.append("runtime resolution was not persisted after repair")
    if runtime_resolution and runtime_resolution.get("runtime_path") != str(cli_path):
        failures.append("runtime resolution did not persist the llama-mtmd CLI path")
    if not commands or commands[0] != LLAMA_MTMD_REPAIR_COMMAND:
        failures.append("simulated native repair did not capture the expected repair command")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "llama-mtmd native runtime repair clears missing ASR GGUF+mmproj runtime evidence through the shared repair plan and persists a CLI runtime resolution."
            if not failures
            else "llama-mtmd native runtime repair contract failed."
        ),
        details={
            "missing_before": list(MISSING_MTMD),
            "missing_after": record.get("after", {}).get("missing", []),
            "install_kind": record.get("before", {}).get("install_kind", ""),
            "repair_action": record.get("repair_action", ""),
            "repair_command": record.get("repair_command", ""),
            "commands": commands,
            "repair_summary": plan.get("summary", {}),
            "runtime_resolution_path": runtime_resolution_path,
            "runtime_resolution": runtime_resolution,
            "backend_probe": backend_probe,
            "runtime_status": runtime_status,
            "failures": failures,
        },
        artifacts=[Path(runtime_resolution_path)] if runtime_resolution_path else [],
    )
