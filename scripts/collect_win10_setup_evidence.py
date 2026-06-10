from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from merge_release_evidence import merge_manual_rows, evidence_rows
    from validate_release_smoke import validate_smoke
except ModuleNotFoundError:
    from scripts.merge_release_evidence import merge_manual_rows, evidence_rows
    from scripts.validate_release_smoke import validate_smoke


ROOT = Path(__file__).resolve().parents[1]
ROW_ID = "win10_existing_python_setup"
WINDOWS_11_BUILD_FLOOR = 22000


def _windows_build_number(version: str) -> int | None:
    for part in reversed(str(version).split(".")):
        if part.isdigit():
            return int(part)
    return None


def validate_win10_row(row: dict) -> list[str]:
    errors: list[str] = []
    if row.get("id") != ROW_ID:
        errors.append(f"row id is {row.get('id')!r}, expected {ROW_ID}")
    if row.get("status") != "pass":
        errors.append(f"{ROW_ID} status is {row.get('status')!r}, expected 'pass'")
    details = row.get("details") if isinstance(row.get("details"), dict) else {}
    platform_info = details.get("platform") if isinstance(details.get("platform"), dict) else {}
    version = str(platform_info.get("version") or "")
    build = _windows_build_number(version)
    if platform_info.get("system") != "Windows":
        errors.append("row platform.system is not Windows")
    if str(platform_info.get("release")) != "10":
        errors.append("row platform.release is not 10")
    if build is None:
        errors.append("row platform.version does not expose a Windows build number")
    elif build >= WINDOWS_11_BUILD_FLOOR:
        errors.append(f"row platform.version build {build} is not below the Windows 11 build floor")
    python_probe = details.get("python_probe") if isinstance(details.get("python_probe"), dict) else {}
    if python_probe.get("python_visible_on_path") is not True:
        errors.append("python/py launcher was not visible on PATH")
    static_contract = details.get("setup_static_contract") if isinstance(details.get("setup_static_contract"), dict) else {}
    if static_contract.get("missing_markers"):
        errors.append("setup static contract has missing markers")
    dry_run = details.get("setup_dry_run_local") if isinstance(details.get("setup_dry_run_local"), dict) else {}
    if dry_run.get("exit_code") != 0:
        errors.append("setup.bat --dry-run --local did not exit 0")
    return errors


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def run_row(output_dir: Path) -> Path:
    workdir = output_dir / ROW_ID
    command = [sys.executable, "qa/runtime_matrix/run_row.py", "--row", ROW_ID, "--workdir", str(workdir)]
    subprocess.run(command, cwd=ROOT, check=True)
    row_path = workdir / "row.json"
    if row_path.exists():
        return row_path
    nested = workdir / ROW_ID / "row.json"
    return nested


def merge_and_validate_smoke(smoke_path: Path, output_smoke: Path, evidence_dir: Path, required_path: Path) -> None:
    smoke = load_json(smoke_path)
    merged = merge_manual_rows(smoke, evidence_rows(evidence_dir, ignore_malformed=True), ignore_unknown=True)
    write_json(output_smoke, merged)
    errors = validate_smoke(
        merged,
        json.loads(required_path.read_text(encoding="utf-8"))["required_rows"],
        require_all_pass=True,
        require_log_hashes=True,
        require_environment_summary=True,
    )
    if errors:
        raise SystemExit("strict release smoke validation failed after Win10 merge:\n  - " + "\n  - ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect and validate the Windows 10 existing-Python setup proof row.")
    parser.add_argument("--output-dir", default=ROOT / "Temp" / "win10_existing_python_setup_evidence", type=Path)
    parser.add_argument("--smoke", type=Path, help="Existing release-smoke JSON to update after the row passes.")
    parser.add_argument("--output-smoke", type=Path, help="Where to write the merged smoke JSON. Defaults to --smoke.")
    parser.add_argument("--required", default=ROOT / "tests" / "fixtures" / "release_required_rows_v2.json", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir
    row_path = run_row(output_dir)
    row = load_json(row_path)
    errors = validate_win10_row(row)
    if errors:
        raise SystemExit("Windows 10 existing-Python evidence is not acceptable:\n  - " + "\n  - ".join(errors))
    if args.smoke:
        merge_and_validate_smoke(args.smoke, args.output_smoke or args.smoke, output_dir, args.required)
    print(row_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
