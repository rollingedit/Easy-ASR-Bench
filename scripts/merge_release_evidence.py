from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _status_rank(row: dict) -> int:
    return {"pass": 4, "fail": 3, "blocked": 2, "not_run": 1}.get(str(row.get("status", "unknown")), 0)


def _row_with_source(row: dict, row_path: Path) -> dict:
    output = dict(row)
    output["evidence_path"] = str(row_path)
    return output


def _merge_duplicate_row(existing: dict, incoming: dict) -> dict:
    variants = list(existing.get("merged_evidence_variants") or [existing])
    variants.extend(incoming.get("merged_evidence_variants") or [incoming])
    best = max(variants, key=_status_rank)
    output = dict(best)
    output["merged_evidence_variants"] = variants
    return output


def _artifact_hash(row: dict, markers: tuple[str, ...]) -> str:
    for artifact in row.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        path = str(artifact.get("path") or "").replace("\\", "/").lower()
        if any(marker in path for marker in markers):
            return str(artifact.get("sha256") or "")
    return ""


def _environment_summary(row: dict, smoke: dict) -> dict:
    environment = row.get("environment")
    if isinstance(environment, dict):
        return dict(environment)
    runner = smoke.get("runner")
    return dict(runner) if isinstance(runner, dict) else {}


def _sanitize_string(value: str) -> str:
    sanitized = re.sub(r"C:[\\/]+Users[\\/]+[^\\/\"\s]+", "%USERPROFILE%", value, flags=re.IGNORECASE)
    sanitized = re.sub(r"[A-Z]:[\\/]+_github[\\/]+[^\\/\"\s<>]+", "%LOCAL_WORKSPACE%", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"(<AccountPassword>)[^<]+(</AccountPassword>)", r"\1<redacted>\2", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"NVIDIA GeForce RTX\s+\d+(?:\s+\w+)?", "NVIDIA CUDA GPU", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"NVIDIA RTX\s+\d+(?:\s+\w+)?", "NVIDIA CUDA GPU", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"Intel\(R\)\s+UHD Graphics\s+\d+", "Intel integrated GPU", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\b\d{1,2}th Gen Intel\(R\)\s+Core\(TM\)\s+i\d+-\d+\w*\b", "Intel CPU", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\bRTX\s+\d+(?:\s+\w+)?", "CUDA dGPU", sanitized, flags=re.IGNORECASE)
    return sanitized


def sanitize_public_evidence(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): sanitize_public_evidence(child) for key, child in value.items()}
    if isinstance(value, list):
        return [sanitize_public_evidence(child) for child in value]
    if isinstance(value, str):
        return _sanitize_string(value)
    return value


def _enrich_release_evidence(row: dict, smoke: dict) -> dict:
    output = dict(row)
    tag = str(smoke.get("tag") or "")
    commit = str(smoke.get("commit") or "")
    if tag:
        output.setdefault("app_version", tag)
    if commit:
        output.setdefault("release_commit", commit)
    environment_summary = _environment_summary(output, smoke)
    if environment_summary:
        output.setdefault("environment_summary", environment_summary)
    logs_hash = _artifact_hash(output, ("/logs/", ".log", "repair_all_safe_last.json"))
    if logs_hash:
        output.setdefault("logs_sha256", logs_hash)
    results_hash = _artifact_hash(output, ("results.json", "scored_report.json", "row.json"))
    if not results_hash:
        evidence_path = output.get("evidence_path")
        if isinstance(evidence_path, str) and evidence_path:
            path = Path(evidence_path)
            if path.exists() and path.is_file():
                results_hash = sha256(path)
    if results_hash:
        output.setdefault("results_sha256", results_hash)
    return sanitize_public_evidence(output)


def evidence_rows(evidence_dir: Path, *, ignore_malformed: bool = False) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row_path in sorted(evidence_dir.rglob("row.json")):
        row = _row_with_source(load_json(row_path), row_path)
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            if ignore_malformed:
                continue
            raise SystemExit(f"Evidence row missing id: {row_path}")
        if row_id in rows:
            rows[row_id] = _merge_duplicate_row(rows[row_id], row)
        else:
            rows[row_id] = row
    return rows


def _status_for(row: object) -> str:
    if isinstance(row, dict):
        return str(row.get("status", "unknown"))
    return str(row)


def _status_counts(rows: list[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = _status_for(row)
        counts[status] = counts.get(status, 0) + 1
    return counts


def _update_matrix_statuses(value: object, evidence: dict[str, dict]) -> object:
    if isinstance(value, dict):
        updated = {}
        for key, child in value.items():
            if isinstance(child, dict):
                updated[key] = _update_matrix_statuses(child, evidence)
            elif key in evidence:
                updated[key] = evidence[key].get("status", child)
            else:
                updated[key] = child
        return updated
    return value


def merge_manual_rows(smoke: dict, evidence: dict[str, dict], *, ignore_unknown: bool = False) -> dict:
    rows = smoke.get("manual_rows")
    if not isinstance(rows, list):
        raise SystemExit("release smoke must contain manual_rows before evidence can be merged")
    output_rows = []
    known_ids = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            output_rows.append(row)
            continue
        row_id = row["id"]
        known_ids.add(row_id)
        output_rows.append(_enrich_release_evidence(evidence[row_id], smoke) if row_id in evidence else row)
    unknown = sorted(set(evidence) - known_ids)
    if unknown:
        if ignore_unknown:
            evidence = {row_id: row for row_id, row in evidence.items() if row_id in known_ids}
            output_rows = []
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                    output_rows.append(row)
                    continue
                row_id = row["id"]
                output_rows.append(_enrich_release_evidence(evidence[row_id], smoke) if row_id in evidence else row)
        else:
            raise SystemExit("Evidence rows are not present in release smoke manual_rows: " + ", ".join(unknown))
    merged = dict(smoke)
    merged["manual_rows"] = output_rows
    if isinstance(smoke.get("manual_matrix"), dict):
        merged["manual_matrix"] = _update_matrix_statuses(smoke["manual_matrix"], evidence)
    merged["manual_row_status_counts"] = _status_counts(output_rows)
    blocked_rows = [
        {
            "id": str(row.get("id", "")),
            "block_reason": str(row.get("block_reason", "")),
            "external_requirement": str(row.get("external_requirement", "")),
        }
        for row in output_rows
        if isinstance(row, dict) and row.get("status") == "blocked"
    ]
    if blocked_rows:
        merged["blocked_rows"] = blocked_rows
    merged.setdefault("notes", [])
    merged["notes"] = list(merged["notes"]) + ["Manual evidence rows were merged from qa/windows_matrix or qa/runtime_matrix row.json files; blocked rows are explicit evidence and are not counted as pass."]
    return merged


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ignore-unknown", action="store_true", help="Ignore evidence rows that are not listed in the release smoke manual rows.")
    args = parser.parse_args()
    smoke = load_json(args.smoke)
    evidence = evidence_rows(args.evidence_dir, ignore_malformed=args.ignore_unknown)
    if not evidence:
        raise SystemExit(f"No row.json evidence files found in {args.evidence_dir}")
    write_json(args.output, merge_manual_rows(smoke, evidence, ignore_unknown=args.ignore_unknown))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
