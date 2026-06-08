from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def evidence_rows(evidence_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row_path in sorted(evidence_dir.rglob("row.json")):
        row = load_json(row_path)
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise SystemExit(f"Evidence row missing id: {row_path}")
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


def merge_manual_rows(smoke: dict, evidence: dict[str, dict]) -> dict:
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
        output_rows.append(evidence.get(row_id, row))
    unknown = sorted(set(evidence) - known_ids)
    if unknown:
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
    args = parser.parse_args()
    smoke = load_json(args.smoke)
    evidence = evidence_rows(args.evidence_dir)
    if not evidence:
        raise SystemExit(f"No row.json evidence files found in {args.evidence_dir}")
    write_json(args.output, merge_manual_rows(smoke, evidence))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
