from __future__ import annotations

from pathlib import Path

from app import repair_plan
from qa.runtime_matrix.common import package_versions, write_row


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id != "media_tools_dependency_repair_contract":
        return write_row(row_id, "fail", evidence_dir, summary=f"Unsupported media tools repair row: {row_id}")
    return _media_tools_dependency_repair_contract(row_id, evidence_dir)


def _media_tools_dependency_repair_contract(row_id: str, evidence_dir: Path) -> dict:
    missing_before = ["imageio_ffmpeg", "ffmpeg executable"]
    installed = {"value": False}
    commands: list[str] = []
    original_dependency_status = repair_plan.dependency_status
    original_acceleration_install_decision = repair_plan.acceleration_install_decision
    original_install_group_for_config = repair_plan.install_group_for_config
    original_missing_modules_for_config = repair_plan.missing_modules_for_config
    original_backend_probe_for_group = repair_plan.backend_probe_for_group

    def fake_dependency_status(config):
        return {
            "media_tools": {
                "available": installed["value"],
                "missing": [] if installed["value"] else list(missing_before),
                "install_kind": "pip",
                "requirement_file": "requirements/core.txt",
                "recovery_command": "python -m pip install -r requirements/core.txt",
                "accelerator_recovery_command": "",
            }
        }

    def fake_install_group_for_config(group, project_root, config, log_path=None):
        if group != "media_tools":
            raise RuntimeError(f"unexpected repair group: {group}")
        command = f"python -m pip install -r {project_root / 'requirements' / 'core.txt'}"
        commands.append(command)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(command + "\n", encoding="utf-8", newline="\n")
        installed["value"] = True
        return {"installed_group": group, "requirement_file": "requirements/core.txt"}

    def fake_missing_modules_for_config(group, config):
        if group == "media_tools":
            return [] if installed["value"] else list(missing_before)
        return []

    def fake_backend_probe_for_group(group, config):
        if group != "media_tools":
            raise RuntimeError(f"unexpected probe group: {group}")
        return {
            "kind": "media_tools_ffmpeg_probe",
            "ok": installed["value"],
            "runtime_status": {
                "available": installed["value"],
                "ffmpeg_path": str(evidence_dir / "Runtime" / "ffmpeg.exe") if installed["value"] else "",
                "ffprobe_path": str(evidence_dir / "Runtime" / "ffprobe.exe") if installed["value"] else "",
                "ffprobe_available": installed["value"],
                "probe_method": "ffprobe" if installed["value"] else "missing",
                "missing": [] if installed["value"] else list(missing_before),
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
    failures = []
    if record.get("affected_dependency_group") != "media_tools":
        failures.append("repair record did not target media_tools")
    if record.get("status") != "repaired":
        failures.append("media_tools repair did not finish as repaired")
    if record.get("repair_action") != "install_missing":
        failures.append("repair action was not install_missing")
    if record.get("after", {}).get("missing") != []:
        failures.append("post-repair missing media tools were not cleared")
    if backend_probe.get("kind") != "media_tools_ffmpeg_probe":
        failures.append("post-repair backend probe was not the media-tools FFmpeg probe")
    if not runtime_status.get("ffmpeg_path"):
        failures.append("post-repair media-tools probe did not record an FFmpeg path")
    if not runtime_resolution_path:
        failures.append("runtime resolution was not persisted after repair")
    if not commands or "requirements\\core.txt" not in commands[0] and "requirements/core.txt" not in commands[0]:
        failures.append("repair command did not target requirements/core.txt")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "Media-tools repair clears missing imageio-ffmpeg/FFmpeg evidence through the shared repair plan and persists an FFmpeg runtime resolution."
            if not failures
            else "Media-tools dependency repair contract failed."
        ),
        details={
            "missing_before": missing_before,
            "missing_after": record.get("after", {}).get("missing", []),
            "repair_action": record.get("repair_action", ""),
            "repair_command": record.get("repair_command", ""),
            "commands": commands,
            "repair_summary": plan.get("summary", {}),
            "runtime_resolution_path": runtime_resolution_path,
            "backend_probe": backend_probe,
            "runtime_status": runtime_status,
            "failures": failures,
            "dependency_versions": package_versions(["imageio-ffmpeg"]),
        },
        artifacts=[Path(runtime_resolution_path)] if runtime_resolution_path else [],
    )
