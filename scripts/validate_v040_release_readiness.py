from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from validate_public_hygiene import scan_history, scan_paths, tracked_files
    from validate_release_smoke import load_json, validate_smoke
except ModuleNotFoundError:
    from scripts.validate_public_hygiene import scan_history, scan_paths, tracked_files
    from scripts.validate_release_smoke import load_json, validate_smoke


ROOT = Path(__file__).resolve().parents[1]


def required_rows(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("rows") or data.get("required_rows") or [])


def validate_readiness(
    *,
    smoke: Path,
    required: Path,
    repo: Path,
    publish_ref: str,
    allow_blocked: bool = False,
    max_history_findings: int = 50,
) -> list[str]:
    errors: list[str] = []
    smoke_data = load_json(smoke)
    rows = required_rows(required)
    errors.extend(
        "strict smoke: " + error
        for error in validate_smoke(
            smoke_data,
            rows,
            require_all_pass=True,
            require_log_hashes=True,
            require_environment_summary=True,
        )
    )
    errors.extend("public tree hygiene: " + error for error in scan_paths(tracked_files(repo)))
    errors.extend("release-smoke hygiene: " + error for error in scan_paths([smoke]))
    errors.extend("public history hygiene: " + error for error in scan_history([publish_ref], root=repo, max_findings=max_history_findings))
    if allow_blocked:
        errors = [error for error in errors if not error.startswith("strict smoke: win10_existing_python_setup status is blocked")]
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate whether v0.4.0 is ready for stable public release.")
    parser.add_argument("--smoke", default=ROOT / "release-smoke-v0.4.0.json", type=Path)
    parser.add_argument("--required", default=ROOT / "tests" / "fixtures" / "release_required_rows_v2.json", type=Path)
    parser.add_argument("--repo", default=ROOT, type=Path, help="Repository whose tracked tree/history would be published.")
    parser.add_argument("--publish-ref", default="main", help="Ref inside --repo intended for publication.")
    parser.add_argument("--max-history-findings", type=int, default=50)
    parser.add_argument(
        "--allow-blocked-win10",
        action="store_true",
        help="For local progress audits only: ignore the known Windows 10 external proof blocker.",
    )
    args = parser.parse_args()
    errors = validate_readiness(
        smoke=args.smoke.resolve(),
        required=args.required.resolve(),
        repo=args.repo.resolve(),
        publish_ref=args.publish_ref,
        allow_blocked=args.allow_blocked_win10,
        max_history_findings=max(1, args.max_history_findings),
    )
    if errors:
        print("v0.4.0 release readiness validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("v0.4.0 release readiness validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
