from __future__ import annotations

from pathlib import Path

from app.dependency_manager import VC_REDIST_REPAIR_COMMAND, visual_cpp_redistributable_status
from qa.runtime_matrix.common import run_command, write_row


def run(row_id: str, evidence_dir: Path, install_deps: bool, _allow_downloads: bool) -> dict:
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
