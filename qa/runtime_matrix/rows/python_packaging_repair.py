from __future__ import annotations

from pathlib import Path

from app import repair_plan
from qa.runtime_matrix.common import write_row


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id != "python_packaging_tools_repair_contract":
        return write_row(row_id, "fail", evidence_dir, summary=f"Unsupported Python packaging repair row: {row_id}")
    return _python_packaging_tools_repair_contract(row_id, evidence_dir)


def _python_packaging_tools_repair_contract(row_id: str, evidence_dir: Path) -> dict:
    missing_before = ["pkg_resources"]
    installed = {"value": False}
    commands: list[str] = []
    original_dependency_status = repair_plan.dependency_status
    original_acceleration_install_decision = repair_plan.acceleration_install_decision
    original_install_group_for_config = repair_plan.install_group_for_config
    original_missing_modules_for_config = repair_plan.missing_modules_for_config
    original_backend_probe_for_group = repair_plan.backend_probe_for_group

    def fake_dependency_status(config):
        return {
            "python_packaging": {
                "available": installed["value"],
                "missing": [] if installed["value"] else list(missing_before),
                "install_kind": "pip",
                "requirement_file": "requirements/python_packaging.txt",
                "recovery_command": "python -m pip install -r requirements/python_packaging.txt",
                "accelerator_recovery_command": "",
            }
        }

    def fake_install_group_for_config(group, project_root, config, log_path=None):
        if group != "python_packaging":
            raise RuntimeError(f"unexpected repair group: {group}")
        command = f"python -m pip install -r {project_root / 'requirements' / 'python_packaging.txt'}"
        commands.append(command)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(command + "\n", encoding="utf-8", newline="\n")
        installed["value"] = True
        return {"installed_group": group, "requirement_file": "requirements/python_packaging.txt"}

    def fake_missing_modules_for_config(group, config):
        if group == "python_packaging":
            return [] if installed["value"] else list(missing_before)
        return []

    def fake_backend_probe_for_group(group, config):
        if group != "python_packaging":
            raise RuntimeError(f"unexpected probe group: {group}")
        return {
            "kind": "python_import_probe",
            "ok": installed["value"],
            "imports": {
                "ok": installed["value"],
                "loaded": [
                    {"module": "pip", "version": "25.0.1"},
                    {"module": "setuptools", "version": "80.10.2"},
                    {"module": "pkg_resources", "version": ""},
                ],
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
    failures = []
    if record.get("affected_dependency_group") != "python_packaging":
        failures.append("repair record did not target python_packaging")
    if record.get("status") != "repaired":
        failures.append("python_packaging repair did not finish as repaired")
    if record.get("repair_action") != "install_missing":
        failures.append("repair action was not install_missing")
    if record.get("after", {}).get("missing") != []:
        failures.append("post-repair missing modules were not cleared")
    if record.get("after", {}).get("backend_probe", {}).get("kind") != "python_import_probe":
        failures.append("post-repair backend probe was not the Python import probe")
    if not record.get("after", {}).get("runtime_resolution_path"):
        failures.append("runtime resolution was not persisted after repair")
    if not commands or "requirements\\python_packaging.txt" not in commands[0] and "requirements/python_packaging.txt" not in commands[0]:
        failures.append("repair command did not target requirements/python_packaging.txt")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "Python packaging repair contract clears missing pkg_resources through the shared repair plan and persists runtime resolution evidence."
            if not failures
            else "Python packaging repair contract failed."
        ),
        details={
            "missing_before": missing_before,
            "missing_after": record.get("after", {}).get("missing", []),
            "repair_command": record.get("repair_command", ""),
            "commands": commands,
            "repair_summary": plan.get("summary", {}),
            "runtime_resolution_path": record.get("after", {}).get("runtime_resolution_path", ""),
            "backend_probe": record.get("after", {}).get("backend_probe", {}),
            "failures": failures,
        },
        artifacts=[Path(record.get("after", {}).get("runtime_resolution_path", ""))] if record.get("after", {}).get("runtime_resolution_path") else [],
    )
