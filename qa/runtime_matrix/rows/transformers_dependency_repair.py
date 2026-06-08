from __future__ import annotations

from pathlib import Path

from app import repair_plan
from qa.runtime_matrix.common import package_versions, write_row


TRANSFORMERS_IMPORTS = ("torch", "transformers", "safetensors", "sentencepiece", "google.protobuf", "torchaudio")


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id != "transformers_cpu_dependency_repair_contract":
        return write_row(row_id, "fail", evidence_dir, summary=f"Unsupported Transformers dependency repair row: {row_id}")
    return _transformers_cpu_dependency_repair_contract(row_id, evidence_dir)


def _transformers_cpu_dependency_repair_contract(row_id: str, evidence_dir: Path) -> dict:
    missing_before = ["sentencepiece", "torchaudio", "transformers>=99.0.0 (installed 4.0.0)"]
    installed = {"value": False}
    commands: list[str] = []
    original_dependency_status = repair_plan.dependency_status
    original_acceleration_install_decision = repair_plan.acceleration_install_decision
    original_install_group_for_config = repair_plan.install_group_for_config
    original_missing_modules_for_config = repair_plan.missing_modules_for_config
    original_backend_probe_for_group = repair_plan.backend_probe_for_group

    def fake_dependency_status(config):
        return {
            "transformers_cpu": {
                "available": installed["value"],
                "missing": [] if installed["value"] else list(missing_before),
                "install_kind": "pip",
                "requirement_file": "requirements/transformers_cpu.txt",
                "recovery_command": "python -m pip install -r requirements/transformers_cpu.txt",
                "accelerator_recovery_command": "",
            }
        }

    def fake_install_group_for_config(group, project_root, config, log_path=None):
        if group != "transformers_cpu":
            raise RuntimeError(f"unexpected repair group: {group}")
        command = f"python -m pip install -r {project_root / 'requirements' / 'transformers_cpu.txt'}"
        commands.append(command)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(command + "\n", encoding="utf-8", newline="\n")
        installed["value"] = True
        return {"installed_group": group, "requirement_file": "requirements/transformers_cpu.txt"}

    def fake_missing_modules_for_config(group, config):
        if group == "transformers_cpu":
            return [] if installed["value"] else list(missing_before)
        return []

    def fake_backend_probe_for_group(group, config):
        if group != "transformers_cpu":
            raise RuntimeError(f"unexpected probe group: {group}")
        return {
            "kind": "python_import_probe",
            "ok": installed["value"],
            "imports": {
                "ok": installed["value"],
                "modules": list(TRANSFORMERS_IMPORTS),
                "loaded": [{"module": module, "version": "fixture"} for module in TRANSFORMERS_IMPORTS],
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
    loaded_modules = [item.get("module") for item in backend_probe.get("imports", {}).get("loaded", [])]
    runtime_resolution_path = record.get("after", {}).get("runtime_resolution_path", "")
    failures = []
    if record.get("affected_dependency_group") != "transformers_cpu":
        failures.append("repair record did not target transformers_cpu")
    if record.get("status") != "repaired":
        failures.append("transformers_cpu repair did not finish as repaired")
    if record.get("repair_action") != "upgrade_outdated":
        failures.append("repair action was not upgrade_outdated for mixed missing/outdated Transformers stack")
    if record.get("after", {}).get("missing") != []:
        failures.append("post-repair missing modules were not cleared")
    if backend_probe.get("kind") != "python_import_probe":
        failures.append("post-repair backend probe was not the Python import probe")
    if set(loaded_modules) != set(TRANSFORMERS_IMPORTS):
        failures.append("post-repair probe did not cover the full Transformers ASR import set")
    if not runtime_resolution_path:
        failures.append("runtime resolution was not persisted after repair")
    if not commands or "requirements\\transformers_cpu.txt" not in commands[0] and "requirements/transformers_cpu.txt" not in commands[0]:
        failures.append("repair command did not target requirements/transformers_cpu.txt")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "Transformers ASR dependency repair covers Torch, Transformers, Safetensors, SentencePiece, protobuf, and Torchaudio through the shared repair plan."
            if not failures
            else "Transformers ASR dependency repair contract failed."
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
            "expected_imports": list(TRANSFORMERS_IMPORTS),
            "loaded_modules": loaded_modules,
            "failures": failures,
            "dependency_versions": package_versions(["torch", "transformers", "safetensors", "sentencepiece", "protobuf", "torchaudio"]),
        },
        artifacts=[Path(runtime_resolution_path)] if runtime_resolution_path else [],
    )
