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
    merged.setdefault("notes", [])
    merged["notes"] = list(merged["notes"]) + ["Manual evidence rows were merged from qa/windows_matrix evidence row.json files."]
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
