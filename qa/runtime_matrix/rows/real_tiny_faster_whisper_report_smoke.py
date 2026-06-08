from __future__ import annotations

import json
from pathlib import Path

from qa.run_real_tiny_model_smoke import main as smoke_main
from qa.runtime_matrix.common import package_versions, write_row


def _latest_smoke_payload() -> tuple[dict, Path]:
    output_root = Path("Temp/real_tiny_model_smoke/Output")
    payloads = sorted(output_root.glob("*/real_tiny_model_smoke.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not payloads:
        raise RuntimeError("real tiny faster-whisper smoke did not produce real_tiny_model_smoke.json")
    return json.loads(payloads[0].read_text(encoding="utf-8")), payloads[0]


def run(row_id: str, evidence_dir: Path, install_deps: bool, _allow_downloads: bool) -> dict:
    import sys

    argv = ["run_real_tiny_model_smoke.py", "--provider", "cpu", "--max-normalized-wer", "0.60"]
    if install_deps:
        argv.append("--install-deps")
    old_argv = sys.argv
    try:
        sys.argv = argv
        smoke_main()
    finally:
        sys.argv = old_argv
    payload, payload_path = _latest_smoke_payload()
    status = "pass" if payload.get("status") == "pass" else "fail"
    return write_row(
        row_id,
        status,
        evidence_dir,
        summary="Real tiny faster-whisper/CTranslate2 app-pipeline smoke completed.",
        details={
            "model": payload.get("model"),
            "metrics": payload.get("metrics"),
            "dependency_versions": package_versions(["faster-whisper", "ctranslate2", "setuptools"]),
            "report_dir": payload.get("report_dir"),
        },
        artifacts=[payload_path],
    )

