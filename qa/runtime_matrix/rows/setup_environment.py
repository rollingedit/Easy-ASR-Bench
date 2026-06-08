from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from qa.runtime_matrix.common import ROOT, write_row


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
        "has_interactive_completion_menu": all(marker in setup for marker in ["Setup complete.", "choice /C RPMIQ", "--download-model-first"]),
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
    is_win11 = details["platform"]["system"] == "Windows" and details["platform"]["release"] == "11"
    python_visible = bool(details["python_probe"]["python_visible_on_path"])
    if failures:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Clean Win11 setup row found broken setup contract markers or dry-run behavior.",
            details={**details, "failures": failures},
        )
    if is_win11 and not python_visible:
        return write_row(
            row_id,
            "pass",
            evidence_dir,
            summary="This environment matches clean Windows 11 with no Python on PATH, and setup dry-run proves bootstrap inputs.",
            details={**details, "failures": failures},
        )
    return write_row(
        row_id,
        "blocked",
        evidence_dir,
        summary="Clean Windows 11 no-Python setup proof requires a VM/Sandbox state this machine does not currently match.",
        block_reason=f"current environment is Windows {details['platform']['release']} with python_visible_on_path={python_visible}",
        external_requirement="Windows 11 VM/Sandbox with no python/py launcher visible on PATH, then run python qa\\runtime_matrix\\run_row.py --row win11_clean_no_python_setup",
        details={**details, "failures": failures},
    )


def _win10_existing_python(row_id: str, evidence_dir: Path) -> dict:
    details = _base_details(evidence_dir)
    failures = []
    if details["setup_static_contract"]["missing_markers"]:
        failures.append("setup.bat missing bootstrap markers")
    if details["setup_dry_run_local"]["exit_code"] != 0:
        failures.append("setup.bat dry-run local failed")
    is_win10 = details["platform"]["system"] == "Windows" and details["platform"]["release"] == "10"
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
        block_reason=f"current environment is Windows {details['platform']['release']} with python_visible_on_path={python_visible}",
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
        "blocked",
        evidence_dir,
        summary="setup.bat double-click flow has a real script contract check, but true double-click proof needs an interactive Windows shell.",
        block_reason="non-interactive runtime matrix cannot click through setup.bat completion choices",
        external_requirement="Interactive Windows shell: double-click setup.bat from a standalone staged folder, verify installer SHA, setup completion menu, and first-run choices",
        details={**details, "failures": failures},
    )


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id == "win11_clean_no_python_setup":
        return _win11_clean_no_python(row_id, evidence_dir)
    if row_id == "win10_existing_python_setup":
        return _win10_existing_python(row_id, evidence_dir)
    if row_id == "setup_double_click_equivalent":
        return _setup_double_click(row_id, evidence_dir)
    return write_row(row_id, "fail", evidence_dir, summary=f"Unsupported setup environment row: {row_id}")
