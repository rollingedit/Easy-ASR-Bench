from __future__ import annotations

from pathlib import Path

from qa.runtime_matrix.common import write_row
from qa.runtime_matrix.registry import ROWS


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    definition = ROWS[row_id]
    reason = (
        "Executable row exists, but this row still needs a dedicated implementation or external runtime fixture "
        f"before it can pass. Required context: {definition.hardware}."
    )
    return write_row(
        row_id,
        "blocked",
        evidence_dir,
        summary=reason,
        block_reason=reason,
        external_requirement=definition.hardware,
        details={"module": definition.module, "description": definition.description},
    )

