from __future__ import annotations

import os
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from qa.runtime_matrix.common import ROOT, write_row

PREBOOTSTRAP_PROBE_ENV = "EASY_ASR_BENCH_PREBOOTSTRAP_PROBE"
WINDOWS_11_BUILD_FLOOR = 22000


def _windows_build_number(version: str) -> int | None:
    for part in reversed(str(version).split(".")):
        if part.isdigit():
            return int(part)
    return None


def _is_windows_11(platform_info: dict) -> bool:
    if platform_info.get("system") != "Windows":
        return False
    build = _windows_build_number(str(platform_info.get("version", "")))
    return str(platform_info.get("release")) == "11" or (build is not None and build >= WINDOWS_11_BUILD_FLOOR)


def _is_windows_10(platform_info: dict) -> bool:
    if platform_info.get("system") != "Windows":
        return False
    build = _windows_build_number(str(platform_info.get("version", "")))
    return str(platform_info.get("release")) == "10" and build is not None and build < WINDOWS_11_BUILD_FLOOR


def _python_probe() -> dict:
    commands = [
        ["python", "--version"],
        ["py", "--version"],
        ["py", "-3.12", "--version"],
    ]
    results = []
    for command in commands:
        executable = shutil.which(command[0])
        result = {"command": command, "resolved": executable}
        if executable:
            try:
                completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
                result.update(
                    {
                        "exit_code": completed.returncode,
                        "stdout": completed.stdout.strip(),
                        "stderr": completed.stderr.strip(),
                    }
                )
            except Exception as exc:
                result.update({"error_type": type(exc).__name__, "error": str(exc)})
        results.append(result)
    return {
        "current_python": sys.executable,
        "path_python_commands": results,
        "python_visible_on_path": any(item.get("resolved") for item in results),
    }


def _load_prebootstrap_probe() -> dict:
    path = os.environ.get(PREBOOTSTRAP_PROBE_ENV, "")
    if not path:
        return {"path": "", "available": False}
    probe_path = Path(path)
    try:
        payload = json.loads(probe_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"path": path, "available": False, "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {"path": path, "available": False, "error": "probe payload is not a JSON object"}
    payload = dict(payload)
    payload["path"] = path
    payload["available"] = True
    return payload


def _setup_static_contract() -> dict:
    setup = (ROOT / "setup.bat").read_text(encoding="utf-8")
    required_markers = [
        ":bootstrap",
        "if exist \"%~dp0app\\main.py\" goto local_setup",
        "Downloading installer script",
        "call :verify_sha",
        "powershell -NoProfile -ExecutionPolicy Bypass -File \"%INSTALLER_PS1%\"",
        "winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements",
        "Setup will download the verified app ZIP",
    ]
    return {
        "required_markers": required_markers,
        "missing_markers": [marker for marker in required_markers if marker not in setup],
        "has_interactive_completion_menu": all(marker in setup for marker in ["Setup complete.", "choice /C RPMIOQ", "--download-model-first"]),
    }


def _dry_run_local(evidence_dir: Path) -> dict:
    env = os.environ.copy()
    temp_dir = evidence_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = str(temp_dir.resolve())
    env["TMP"] = str(temp_dir.resolve())
    completed = subprocess.run(["cmd", "/c", "setup.bat", "--dry-run", "--local"], cwd=ROOT, text=True, capture_output=True, timeout=180, env=env)
    return {
        "command": ["cmd", "/c", "setup.bat", "--dry-run", "--local"],
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-6000:],
        "stderr_tail": completed.stderr[-6000:],
    }


def _base_details(evidence_dir: Path) -> dict:
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python_probe": _python_probe(),
        "prebootstrap_probe": _load_prebootstrap_probe(),
        "setup_static_contract": _setup_static_contract(),
        "setup_dry_run_local": _dry_run_local(evidence_dir),
    }


def _win11_clean_no_python(row_id: str, evidence_dir: Path) -> dict:
    details = _base_details(evidence_dir)
    failures = []
    if details["setup_static_contract"]["missing_markers"]:
        failures.append("setup.bat missing clean-bootstrap markers")
    if details["setup_dry_run_local"]["exit_code"] != 0:
        failures.append("setup.bat dry-run local failed")
    is_win11 = _is_windows_11(details["platform"])
    python_visible = bool(details["python_probe"]["python_visible_on_path"])
    prebootstrap = details["prebootstrap_probe"]
    prebootstrap_proves_no_python = bool(
        prebootstrap.get("available")
        and prebootstrap.get("system") == "Windows"
        and str(prebootstrap.get("release")) == "11"
        and prebootstrap.get("python_visible_on_path") is False
    )
    if failures:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Clean Win11 setup row found broken setup contract markers or dry-run behavior.",
            details={**details, "failures": failures},
        )
    if is_win11 and (not python_visible or prebootstrap_proves_no_python):
        return write_row(
            row_id,
            "pass",
            evidence_dir,
            summary="Clean Windows 11 no-Python bootstrap inputs were proven, and setup dry-run proves bootstrap inputs.",
            details={**details, "failures": failures},
        )
    return write_row(
        row_id,
        "blocked",
        evidence_dir,
        summary="Clean Windows 11 no-Python setup proof requires a VM/Sandbox state this machine does not currently match.",
        block_reason=f"current environment is Windows {details['platform']['release']} with python_visible_on_path={python_visible}",
        external_requirement=(
            "Windows 11 VM/Sandbox with no python/py launcher visible before setup; capture a pre-bootstrap probe JSON, set "
            f"{PREBOOTSTRAP_PROBE_ENV} to it after setup installs Python, then run python qa\\runtime_matrix\\run_row.py --row win11_clean_no_python_setup"
        ),
        details={**details, "failures": failures},
    )


def _win10_existing_python(row_id: str, evidence_dir: Path) -> dict:
    details = _base_details(evidence_dir)
    failures = []
    if details["setup_static_contract"]["missing_markers"]:
        failures.append("setup.bat missing bootstrap markers")
    if details["setup_dry_run_local"]["exit_code"] != 0:
        failures.append("setup.bat dry-run local failed")
    is_win10 = _is_windows_10(details["platform"])
    python_visible = bool(details["python_probe"]["python_visible_on_path"])
    if failures:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Win10 existing-Python setup row found broken setup contract markers or dry-run behavior.",
            details={**details, "failures": failures},
        )
    if is_win10 and python_visible:
        return write_row(
            row_id,
            "pass",
            evidence_dir,
            summary="This environment matches Windows 10 with an existing Python launcher, and setup dry-run proves bootstrap inputs.",
            details={**details, "failures": failures},
        )
    return write_row(
        row_id,
        "blocked",
        evidence_dir,
        summary="Windows 10 existing-Python setup proof requires a VM state this machine does not currently match.",
        block_reason=(
            f"current environment is Windows release={details['platform']['release']} "
            f"version={details['platform']['version']} with python_visible_on_path={python_visible}"
        ),
        external_requirement="Windows 10 VM with Python 3.10-3.14 already visible, then run python qa\\runtime_matrix\\run_row.py --row win10_existing_python_setup",
        details={**details, "failures": failures},
    )


def _setup_double_click(row_id: str, evidence_dir: Path) -> dict:
    details = _base_details(evidence_dir)
    failures = []
    markers = details["setup_static_contract"]
    if markers["missing_markers"]:
        failures.append("setup.bat missing double-click/bootstrap markers")
    if not markers["has_interactive_completion_menu"]:
        failures.append("setup.bat missing post-install interactive completion menu")
    if details["setup_dry_run_local"]["exit_code"] != 0:
        failures.append("setup.bat dry-run local failed")
    if failures:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Double-click setup contract markers failed.",
            details={**details, "failures": failures},
        )
    return write_row(
        row_id,
        "pass",
        evidence_dir,
        summary="setup.bat double-click equivalent passed: bootstrap markers, SHA verification path, completion-menu markers, and non-destructive local dry-run are valid.",
        details={**details, "failures": failures},
    )


def _first_run_smoke_json(row_id: str, evidence_dir: Path) -> dict:
    config = {
        "folders": {
            "models": str(evidence_dir / "Models"),
            "input": str(evidence_dir / "Input"),
            "output": str(evidence_dir / "Output"),
            "temp": str(evidence_dir / "Temp"),
            "logs": str(evidence_dir / "Logs"),
            "cache": str(evidence_dir / "Cache"),
        }
    }
    config_path = evidence_dir / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")
    command = [sys.executable, "-m", "app.main", "--config", str(config_path), "--first-run-smoke", "--json"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    smoke_path = evidence_dir / "first-run-smoke.json"
    payload = {}
    failures: list[str] = []
    if completed.returncode != 0:
        failures.append(f"first-run smoke command exited {completed.returncode}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        failures.append(f"first-run smoke stdout was not JSON: {exc}")
    if payload:
        smoke_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        if payload.get("schema") != "easy_asr_bench.first_run_smoke.v1":
            failures.append("first-run smoke schema is missing or invalid")
        if payload.get("repair_plan_schema") != "easy_asr_bench.repair_plan.v1":
            failures.append("first-run smoke did not include repair-plan evidence")
        if payload.get("repair_command") != "setup.bat --doctor --repair-all-safe":
            failures.append("first-run smoke repair command is missing or wrong")
        if payload.get("model_layout_repair_command") != "setup.bat --doctor --repair-model-layouts --allow-downloads":
            failures.append("first-run smoke model-layout repair command is missing or wrong")
        if payload.get("doctor_command") != "setup.bat --doctor --repair-plan":
            failures.append("first-run smoke doctor command is missing or wrong")
        if payload.get("real_smoke_command") != "setup.bat --doctor --validate-real-smoke":
            failures.append("first-run smoke real-smoke command is missing or wrong")
        actions = set(payload.get("available_actions") or [])
        missing_actions = {"download_recommended_baseline", "paste_hugging_face_link", "open_models_folder", "open_input_folder"} - actions
        if missing_actions:
            failures.append("first-run smoke missing available actions: " + ", ".join(sorted(missing_actions)))
        if payload.get("dead_end") is not False:
            failures.append("first-run smoke reported a dead end")
    return write_row(
        row_id,
        "fail" if failures else "pass",
        evidence_dir,
        summary=(
            "First-run smoke emits actionable JSON with repair-plan, doctor, real-smoke, and next-action evidence."
            if not failures
            else "First-run smoke JSON validation failed."
        ),
        details={
            "command": command,
            "exit_code": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
            "payload": payload,
            "failures": failures,
        },
        artifacts=[config_path, smoke_path],
    )


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id == "win11_clean_no_python_setup":
        return _win11_clean_no_python(row_id, evidence_dir)
    if row_id == "win10_existing_python_setup":
        return _win10_existing_python(row_id, evidence_dir)
    if row_id == "setup_double_click_equivalent":
        return _setup_double_click(row_id, evidence_dir)
    if row_id == "first_run_smoke_json":
        return _first_run_smoke_json(row_id, evidence_dir)
    return write_row(row_id, "fail", evidence_dir, summary=f"Unsupported setup environment row: {row_id}")
