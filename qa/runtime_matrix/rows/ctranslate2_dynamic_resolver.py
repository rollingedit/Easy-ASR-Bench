from __future__ import annotations

from pathlib import Path

from app import dependency_manager as dm
from app import main as app_main
from app.config import load_config
from app.dependency_manager import missing_modules_for_config
from app.adapters.base import ModelCandidate
from qa.runtime_matrix.common import package_versions, write_row
from qa.runtime_matrix.rows.real_tiny_faster_whisper_report_smoke import run as run_smoke


def _pkg_resources_repair_contract(row_id: str, evidence_dir: Path) -> dict:
    from app.adapters import faster_whisper_asr

    evidence_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = evidence_dir / "Models" / "tiny-ct2"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate = ModelCandidate(
        candidate_id="faster_whisper_pkg_resources_contract",
        display_name="faster-whisper pkg_resources repair contract",
        family_name="faster-whisper",
        backend="faster-whisper",
        container_format="ctranslate2",
        task="automatic-speech-recognition",
        precision="int8",
        quantization_label="int8",
        path=candidate_dir,
        adapter_name="faster_whisper",
        runnable=True,
    )
    config = {
        "runtime": {"provider": "cpu", "prefer_gpu": False, "fallback_to_cpu": True},
        "folders": {"logs": str(evidence_dir / "Logs")},
    }
    commands: list[list[str]] = []
    probe_calls: list[dict] = []

    class Completed:
        returncode = 0
        stdout = "simulated faster-whisper requirement repair\n"

    original = {
        "dm_missing_import_modules": dm._missing_import_modules,
        "dm_requirement_version_issues": dm.requirement_version_issues,
        "main_subprocess_run": app_main.subprocess.run,
        "runtime_choices": faster_whisper_asr.faster_whisper_runtime_choices,
        "probe": faster_whisper_asr.probe_faster_whisper_load,
    }

    def fake_missing_import_modules(metadata) -> list[str]:
        return ["pkg_resources"] if "pkg_resources" in metadata.modules else []

    def fake_subprocess_run(command, cwd=None, text=None, stdout=None, stderr=None, timeout=None):
        commands.append([str(item) for item in command])
        return Completed()

    def fake_runtime_choices(candidate, runtime_config):
        return None, "cpu", "int8", "int8", []

    def fake_probe(path, device, compute_type) -> str:
        probe_calls.append({"path": str(path), "device": device, "compute_type": compute_type})
        return ""

    try:
        dm._missing_import_modules = fake_missing_import_modules
        dm.requirement_version_issues = lambda requirement_files, ignored_packages=None: []
        app_main.subprocess.run = fake_subprocess_run
        faster_whisper_asr.faster_whisper_runtime_choices = fake_runtime_choices
        faster_whisper_asr.probe_faster_whisper_load = fake_probe
        missing_before = dm.missing_modules_for_config("faster_whisper", config)
        repair_error = app_main._repair_faster_whisper_native_stack(candidate, config, "ModuleNotFoundError: No module named 'pkg_resources'")
        missing_after = []
    finally:
        dm._missing_import_modules = original["dm_missing_import_modules"]
        dm.requirement_version_issues = original["dm_requirement_version_issues"]
        app_main.subprocess.run = original["main_subprocess_run"]
        faster_whisper_asr.faster_whisper_runtime_choices = original["runtime_choices"]
        faster_whisper_asr.probe_faster_whisper_load = original["probe"]

    log_paths = sorted((evidence_dir / "Logs").glob("dependency_install_faster_whisper_native_compatibility_*.log"))
    repair_log = log_paths[-1] if log_paths else evidence_dir / "Logs" / "missing.log"
    expected_requirement = str(Path.cwd() / "requirements" / "faster_whisper.txt")
    failures = []
    if "pkg_resources" not in missing_before:
        failures.append("pkg_resources missing marker was not detected before repair")
    if repair_error:
        failures.append(f"native repair returned an error: {repair_error}")
    if missing_after:
        failures.append("missing markers remained after simulated repair: " + ", ".join(missing_after))
    if not commands:
        failures.append("native repair did not run a dependency command")
    elif commands[0][-2:] != ["-r", expected_requirement]:
        failures.append("first repair command did not reinstall requirements/faster_whisper.txt")
    if commands and "--force-reinstall" not in commands[0]:
        failures.append("first repair command did not force reinstall the bounded requirement set")
    if not probe_calls:
        failures.append("native load probe was not rerun after repair")
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "faster-whisper/CTranslate2 pkg_resources failure is detected and repaired through the bounded product native-stack repair path."
            if not failures
            else "faster-whisper/CTranslate2 pkg_resources repair contract failed."
        ),
        details={
            "simulated_initial_error": "ModuleNotFoundError: No module named 'pkg_resources'",
            "missing_before": missing_before,
            "missing_after": missing_after,
            "repair_error": repair_error,
            "commands": commands,
            "probe_calls": probe_calls,
            "expected_requirement": expected_requirement,
            "failures": failures,
            "dependency_versions": package_versions(["faster-whisper", "ctranslate2", "setuptools"]),
        },
        artifacts=[repair_log],
    )


def run(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    if row_id == "faster_whisper_pkg_resources_repair":
        return _pkg_resources_repair_contract(row_id, evidence_dir)
    config = load_config(Path("config.json"))
    config["runtime"]["provider"] = "cpu"
    config["runtime"]["prefer_gpu"] = False
    missing = missing_modules_for_config("faster_whisper", config)
    if missing and not install_deps:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="faster-whisper/CTranslate2 runtime is not currently import-complete; run with --install-deps to repair and smoke.",
            block_reason=", ".join(missing),
            external_requirement="network/package install approval for faster_whisper dependency group",
            details={"missing": missing, "dependency_versions": package_versions(["faster-whisper", "ctranslate2", "setuptools"])},
        )
    return run_smoke(row_id, evidence_dir, install_deps, allow_downloads)
