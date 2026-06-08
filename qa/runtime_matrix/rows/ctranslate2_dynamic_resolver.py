from __future__ import annotations

from pathlib import Path

from app.config import load_config
from app.dependency_manager import missing_modules_for_config
from qa.runtime_matrix.common import package_versions, write_row
from qa.runtime_matrix.rows.real_tiny_faster_whisper_report_smoke import run as run_smoke


def run(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
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

