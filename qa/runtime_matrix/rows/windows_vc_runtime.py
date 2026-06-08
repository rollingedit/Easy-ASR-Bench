from __future__ import annotations

from pathlib import Path

from app.dependency_manager import VC_REDIST_REPAIR_COMMAND, visual_cpp_redistributable_status
from qa.runtime_matrix.common import run_command, write_row


def _vc_redist_repair_contract(row_id: str, evidence_dir: Path) -> dict:
    states = [
        {
            "installed": False,
            "version": "",
            "source": "simulated_missing",
            "repair_command": VC_REDIST_REPAIR_COMMAND,
            "details": ["simulated clean Windows state with VC++ Redistributable missing"],
        },
        {
            "installed": True,
            "version": "14.51.36231",
            "source": "simulated_winget_after_repair",
            "repair_command": VC_REDIST_REPAIR_COMMAND,
            "details": ["simulated VC++ Redistributable visible after winget repair"],
        },
    ]
    commands: list[list[str]] = []

    def fake_status(*, include_winget: bool = False) -> dict:
        return states[0] if not commands else states[1]

    def fake_run_command(command: list[str], *, timeout: int = 300, **_kwargs) -> dict:
        commands.append(list(command))
        return {"command": list(command), "exit_code": 0, "stdout_tail": "simulated winget VC++ repair\n", "stderr_tail": ""}

    original_status = globals()["visual_cpp_redistributable_status"]
    original_run_command = globals()["run_command"]
    try:
        globals()["visual_cpp_redistributable_status"] = fake_status
        globals()["run_command"] = fake_run_command
        row = run("windows_vc_runtime", evidence_dir, True, False)
    finally:
        globals()["visual_cpp_redistributable_status"] = original_status
        globals()["run_command"] = original_run_command

    expected = [
        "winget",
        "install",
        "-e",
        "--id",
        "Microsoft.VCRedist.2015+.x64",
        "--accept-package-agreements",
        "--accept-source-agreements",
    ]
    failures = list(row.get("details", {}).get("failures", []))
    if row.get("status") != "pass":
        failures.append("simulated VC++ repair row did not pass")
    if commands != [expected]:
        failures.append("VC++ repair did not run the expected winget command with agreement flags")
    details = {
        **row.get("details", {}),
        "simulated_missing_before": states[0],
        "simulated_installed_after": states[1],
        "commands": commands,
        "expected_command": expected,
        "failures": failures,
    }
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "VC++ Redistributable repair contract invokes the winget repair command with required agreement flags and reprobes successfully."
            if not failures
            else "VC++ Redistributable repair contract failed."
        ),
        details=details,
    )


def run(row_id: str, evidence_dir: Path, install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id == "windows_vc_runtime_repair_contract":
        return _vc_redist_repair_contract(row_id, evidence_dir)
    status = visual_cpp_redistributable_status(include_winget=True)
    repair_result = None
    if not status["installed"] and install_deps:
        repair_result = run_command(
            [
                "winget",
                "install",
                "-e",
                "--id",
                "Microsoft.VCRedist.2015+.x64",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ],
            timeout=600,
        )
        status = visual_cpp_redistributable_status(include_winget=True)

    details = {
        "visual_cpp_redistributable": status,
        "repair_command": VC_REDIST_REPAIR_COMMAND,
    }
    if repair_result is not None:
        details["repair_result"] = repair_result

    if status["installed"]:
        return write_row(
            row_id,
            "pass",
            evidence_dir,
            summary="Microsoft Visual C++ 2015-2022 Redistributable x64 is installed for native ASR backends.",
            details=details,
        )
    return write_row(
        row_id,
        "blocked",
        evidence_dir,
        summary="Microsoft Visual C++ 2015-2022 Redistributable x64 is not installed or could not be detected.",
        block_reason="VC++ Redistributable x64 missing or undetected",
        external_requirement=VC_REDIST_REPAIR_COMMAND,
        details=details,
    )
