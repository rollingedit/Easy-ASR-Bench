from __future__ import annotations

from pathlib import Path

from app import dependency_manager as dm
from app import main as app_main
from app.config import load_config
from app.dependency_manager import missing_modules_for_config
from app.adapters.base import ModelCandidate
from qa.runtime_matrix.common import package_versions, write_row
from qa.runtime_matrix.rows.real_tiny_faster_whisper_report_smoke import run as run_smoke


def _contract_candidate(evidence_dir: Path, candidate_id: str) -> ModelCandidate:
    candidate_dir = evidence_dir / "Models" / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    return ModelCandidate(
        candidate_id=candidate_id,
        display_name=candidate_id.replace("_", " "),
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


def _pkg_resources_repair_contract(row_id: str, evidence_dir: Path) -> dict:
    from app.adapters import faster_whisper_asr

    evidence_dir.mkdir(parents=True, exist_ok=True)
    candidate = _contract_candidate(evidence_dir, "faster_whisper_pkg_resources_contract")
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


def _ctranslate2_candidate_fallback_contract(row_id: str, evidence_dir: Path) -> dict:
    from app.adapters import faster_whisper_asr

    evidence_dir.mkdir(parents=True, exist_ok=True)
    candidate = _contract_candidate(evidence_dir, "faster_whisper_ctranslate2_candidate_contract")
    config = {
        "runtime": {"provider": "cpu", "prefer_gpu": False, "fallback_to_cpu": True},
        "folders": {"logs": str(evidence_dir / "Logs")},
    }
    commands: list[list[str]] = []
    probe_calls: list[dict] = []

    original = {
        "main_subprocess_run": app_main.subprocess.run,
        "runtime_choices": faster_whisper_asr.faster_whisper_runtime_choices,
        "probe": faster_whisper_asr.probe_faster_whisper_load,
    }

    def fake_subprocess_run(command, cwd=None, text=None, stdout=None, stderr=None, timeout=None):
        command = [str(item) for item in command]
        commands.append(command)
        if command[:4] == [__import__("sys").executable, "-m", "pip", "index"]:
            return __import__("subprocess").CompletedProcess(
                command,
                0,
                "Available versions: 4.10.0, 4.9.0, 4.8.0, 4.7.2, 4.4.0, 4.3.1\n",
            )
        return __import__("subprocess").CompletedProcess(command, 0, "simulated install\n")

    probe_results = iter(["requirement set still broken", "ctranslate2 4.8.0 still broken", ""])

    def fake_runtime_choices(candidate, runtime_config):
        return None, "cpu", "int8", "int8", []

    def fake_probe(path, device, compute_type) -> str:
        result = next(probe_results)
        probe_calls.append({"path": str(path), "device": device, "compute_type": compute_type, "result": result or "pass"})
        return result

    try:
        app_main.subprocess.run = fake_subprocess_run
        faster_whisper_asr.faster_whisper_runtime_choices = fake_runtime_choices
        faster_whisper_asr.probe_faster_whisper_load = fake_probe
        repair_error = app_main._repair_faster_whisper_native_stack(candidate, config, "initial CTranslate2 native load failure")
    finally:
        app_main.subprocess.run = original["main_subprocess_run"]
        faster_whisper_asr.faster_whisper_runtime_choices = original["runtime_choices"]
        faster_whisper_asr.probe_faster_whisper_load = original["probe"]

    log_paths = sorted((evidence_dir / "Logs").glob("dependency_install_faster_whisper_native_compatibility_*.log"))
    repair_log = log_paths[-1] if log_paths else evidence_dir / "Logs" / "missing.log"
    install_specs = [part for command in commands for part in command if part.startswith("ctranslate2==")]
    ignored_versions = [version for version in ["4.10.0", "4.9.0", "4.3.1"] if f"ctranslate2=={version}" in install_specs]
    failures = []
    if repair_error:
        failures.append(f"native repair returned an error: {repair_error}")
    if install_specs != ["ctranslate2==4.8.0", "ctranslate2==4.7.2"]:
        failures.append("candidate install order was not newest in-range first, then fallback: " + ", ".join(install_specs))
    if ignored_versions:
        failures.append("out-of-range CTranslate2 versions were attempted: " + ", ".join(ignored_versions))
    if len(probe_calls) != 3:
        failures.append(f"expected three native probes, got {len(probe_calls)}")
    elif probe_calls[-1]["result"] != "pass":
        failures.append("fallback candidate did not pass the native probe")
    if not any(command[:4] == [__import__("sys").executable, "-m", "pip", "index"] for command in commands):
        failures.append("repair did not discover candidates through pip index")
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "faster-whisper/CTranslate2 native repair tries bounded discovered candidates and falls back only after observed probe failures."
            if not failures
            else "faster-whisper/CTranslate2 candidate fallback repair contract failed."
        ),
        details={
            "simulated_initial_error": "initial CTranslate2 native load failure",
            "commands": commands,
            "candidate_install_specs": install_specs,
            "ignored_out_of_range_versions": ["4.10.0", "4.9.0", "4.3.1"],
            "probe_calls": probe_calls,
            "repair_error": repair_error,
            "failures": failures,
            "dependency_versions": package_versions(["faster-whisper", "ctranslate2", "setuptools"]),
        },
        artifacts=[repair_log],
    )


def run(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    if row_id == "faster_whisper_pkg_resources_repair":
        return _pkg_resources_repair_contract(row_id, evidence_dir)
    if row_id == "faster_whisper_ctranslate2_candidate_fallback_repair":
        return _ctranslate2_candidate_fallback_contract(row_id, evidence_dir)
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
