from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_manual_rows(smoke: dict) -> dict[str, object]:
    rows = smoke.get("manual_rows")
    if isinstance(rows, list):
        output = {}
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                output[row["id"]] = row
        return output
    matrix = smoke.get("manual_matrix", {})
    output: dict[str, object] = {}

    def walk(prefix: str, value):
        if isinstance(value, dict):
            if "status" in value and prefix:
                output[prefix] = value
                return
            for key, child in value.items():
                walk(key if not prefix else f"{prefix}.{key}", child)
        elif prefix:
            output[prefix] = value

    walk("", matrix)
    return output


def row_status(row: object) -> str:
    if isinstance(row, dict):
        return str(row.get("status", "missing"))
    return str(row)


def validate_smoke(
    smoke: dict,
    required_rows: list[str],
    *,
    require_all_pass: bool = False,
    require_log_hashes: bool = False,
    require_environment_summary: bool = False,
) -> list[str]:
    errors: list[str] = []
    if smoke.get("schema") not in {"easy_asr_bench.release_smoke.v1", "easy_asr_bench.release_smoke.v2"}:
        errors.append("release smoke has an unexpected schema")
    rows = flatten_manual_rows(smoke)
    for row_id in required_rows:
        if row_id not in rows:
            errors.append(f"missing required smoke row: {row_id}")
            continue
        row = rows[row_id]
        status = row_status(row)
        if require_all_pass and status != "pass":
            errors.append(f"{row_id} status is {status}, expected pass")
        if isinstance(row, dict):
            if status == "blocked":
                if not row.get("block_reason"):
                    errors.append(f"{row_id} blocked row is missing block_reason")
                if not row.get("external_requirement"):
                    errors.append(f"{row_id} blocked row is missing external_requirement")
            if require_all_pass:
                if not row.get("app_version"):
                    errors.append(f"{row_id} is missing app_version")
                if not row.get("release_commit"):
                    errors.append(f"{row_id} is missing release_commit")
            if require_log_hashes and not (row.get("logs_sha256") or row.get("results_sha256")):
                errors.append(f"{row_id} is missing logs_sha256/results_sha256")
            if require_environment_summary and not isinstance(row.get("environment_summary"), dict):
                errors.append(f"{row_id} is missing environment_summary")
        elif require_log_hashes or require_environment_summary:
            errors.append(f"{row_id} must be an object row for strict evidence validation")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", required=True, type=Path)
    parser.add_argument("--required", required=True, type=Path)
    parser.add_argument("--require-all-pass", action="store_true")
    parser.add_argument("--require-log-hashes", action="store_true")
    parser.add_argument("--require-environment-summary", action="store_true")
    args = parser.parse_args()
    required = load_json(args.required).get("rows", [])
    errors = validate_smoke(
        load_json(args.smoke),
        list(required),
        require_all_pass=args.require_all_pass,
        require_log_hashes=args.require_log_hashes,
        require_environment_summary=args.require_environment_summary,
    )
    if errors:
        print("release smoke validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("release smoke validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
