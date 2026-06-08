from __future__ import annotations

from pathlib import Path

from app.model_scanner import scan_models
from qa.runtime_matrix.common import write_row


def _candidate_summary(candidates: list) -> list[dict]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            "display_name": candidate.display_name,
            "adapter_name": candidate.adapter_name,
            "path": str(candidate.path),
            "runnable": candidate.runnable,
            "missing_files": candidate.missing_files,
            "warnings": candidate.warnings,
        }
        for candidate in candidates
    ]


def _write_faster_whisper_fixture(root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for name in ["model.bin", "config.json", "tokenizer.json"]:
        path = root / name
        path.write_text("{}", encoding="utf-8", newline="\n")
        artifacts.append(path)
    return artifacts


def _empty_row(row_id: str, evidence_dir: Path) -> dict:
    models_root = evidence_dir / "Models"
    models_root.mkdir(parents=True, exist_ok=True)
    runnable, unsupported = scan_models(models_root)
    ok = not runnable and not unsupported
    return write_row(
        row_id,
        "pass" if ok else "fail",
        evidence_dir,
        summary=(
            "Empty Models folder scans without producing runnable or unsupported model candidates."
            if ok
            else "Empty Models folder unexpectedly produced model candidates."
        ),
        details={
            "models_root": str(models_root),
            "runnable": _candidate_summary(runnable),
            "unsupported": _candidate_summary(unsupported),
        },
    )


def _nested_row(row_id: str, evidence_dir: Path) -> dict:
    models_root = evidence_dir / "Models"
    artifacts = []
    for rel in ["Group A/same", "Group B/same"]:
        artifacts.extend(_write_faster_whisper_fixture(models_root / rel))
    runnable, unsupported = scan_models(models_root)
    all_candidates = [*runnable, *unsupported]
    ids = [candidate.candidate_id for candidate in all_candidates]
    ok = (
        len([candidate for candidate in runnable if candidate.adapter_name == "faster_whisper"]) == 2
        and len(ids) == len(set(ids))
        and any("group_a" in candidate_id for candidate_id in ids)
        and any("group_b" in candidate_id for candidate_id in ids)
    )
    return write_row(
        row_id,
        "pass" if ok else "fail",
        evidence_dir,
        summary=(
            "Nested model folders with duplicate leaf names scan as distinct candidates with unique IDs."
            if ok
            else "Nested model folders did not produce unique candidate IDs."
        ),
        details={
            "models_root": str(models_root),
            "candidate_ids": ids,
            "runnable": _candidate_summary(runnable),
            "unsupported": _candidate_summary(unsupported),
        },
        artifacts=artifacts,
    )


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id in {"empty_models_folder", "empty_models"}:
        return _empty_row(row_id, evidence_dir)
    if row_id in {"nested_models_folders", "nested_models_scan"}:
        return _nested_row(row_id, evidence_dir)
    return write_row(row_id, "fail", evidence_dir, summary=f"Unhandled model folder scanner row: {row_id}")
