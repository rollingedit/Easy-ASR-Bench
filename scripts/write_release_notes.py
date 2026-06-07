from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def changelog_section(tag: str) -> list[str]:
    version = tag[1:] if tag.startswith("v") else tag
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    marker = f"## v{version}"
    if marker not in text:
        return []
    section = text.split(marker, 1)[1]
    if "\n## " in section:
        section = section.split("\n## ", 1)[0]
    changes = []
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if changes and line.endswith(":") and not line.startswith("- "):
            break
        if line.startswith("- "):
            changes.append(line)
    return changes


def asset_hash_lines() -> list[str]:
    checksums_path = ROOT / "installer" / "checksums.json"
    if not checksums_path.exists():
        return ["- Asset hashes were not available in `installer/checksums.json`."]
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    return [f"- `{name}`: `{digest}`" for name, digest in sorted(checksums.get("files", {}).items())]


def current_commit() -> str:
    try:
        completed = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True)
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def smoke_sections(smoke_path: Path | None) -> tuple[list[str], list[str]]:
    if smoke_path is None or not smoke_path.exists():
        return [], ["- No release smoke artifact was provided to release-note generation."]
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    rows = smoke.get("manual_rows") or []
    if not rows:
        matrix = smoke.get("manual_matrix", {})
        for key, value in matrix.items():
            if isinstance(value, dict):
                rows.extend({"id": nested_key, "status": nested_value} for nested_key, nested_value in value.items())
            else:
                rows.append({"id": key, "status": value})
    passed = []
    not_verified = []
    for row in rows:
        row_id = str(row.get("id", "unknown"))
        status = str(row.get("status", "unknown"))
        line = f"- `{row_id}`: {status}"
        if status == "pass":
            passed.append(line)
        else:
            not_verified.append(line)
    return passed or ["- No manual Windows/model/provider/media rows are marked pass in the smoke artifact."], not_verified


def smoke_rows(smoke_path: Path | None) -> list[dict]:
    if smoke_path is None or not smoke_path.exists():
        return []
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    rows = smoke.get("manual_rows") or []
    if rows:
        return [row for row in rows if isinstance(row, dict)]
    flattened: list[dict] = []
    for key, value in (smoke.get("manual_matrix") or {}).items():
        if isinstance(value, dict):
            flattened.extend({"id": nested_key, "status": nested_value} for nested_key, nested_value in value.items())
        else:
            flattened.append({"id": key, "status": value})
    return flattened


def automated_check_lines(smoke_path: Path | None) -> list[str]:
    if smoke_path is None or not smoke_path.exists():
        return ["- No release smoke artifact was provided."]
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    checks = smoke.get("checks") or []
    if not checks:
        return ["- No automated check rows were recorded in the release smoke artifact."]
    lines = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name", "unknown_check"))
        status = str(check.get("status", "unknown"))
        command = str(check.get("command", "")).strip()
        suffix = f" - `{command}`" if command else ""
        lines.append(f"- `{name}`: {status}{suffix}")
    return lines or ["- No automated check rows were recorded in the release smoke artifact."]


def release_status_line(smoke_path: Path | None) -> str:
    rows = smoke_rows(smoke_path)
    if not rows:
        return "Release status: smoke evidence missing; treat this build as unverified."
    unverified = [row for row in rows if str(row.get("status", "unknown")) != "pass"]
    if unverified:
        return (
            "Release status: automated packaging checks may be present, but this is not an all-pass "
            "manual smoke release because required rows remain unverified."
        )
    return "Release status: required manual smoke rows are marked pass in the smoke artifact."


def write_notes(tag: str, output: Path, smoke_path: Path | None = None) -> None:
    changes = changelog_section(tag)
    if not changes:
        changes = ["- Maintenance release with validated packaging and installer updates."]
    smoke_passed, smoke_not_verified = smoke_sections(smoke_path)
    body = [
        f"# Easy ASR Bench {tag}",
        "",
        release_status_line(smoke_path),
        "",
        "## What changed",
        "",
        *changes,
        "",
        "## Automated Packaging Checks",
        "",
        f"- Built from commit: `{current_commit()}`.",
        "- Public setup verification path: `setup.bat --dry-run --verify-release`.",
        "- GitHub Actions Release Gate must pass before a release should be promoted.",
        *automated_check_lines(smoke_path),
        "",
        "## Manual Smoke Rows Marked Pass",
        "",
        *smoke_passed,
        "",
        "## Not Verified In Release Smoke",
        "",
        *smoke_not_verified,
        "",
        "## Release assets",
        "",
        *asset_hash_lines(),
        "",
        "## Known limits",
        "",
        "- Optional model dependency groups install only when needed.",
        "- GPU/VRAM metrics require a CUDA-capable runtime; CPU runs report VRAM as `null`.",
        "- Unsafe pickle-backed `.pt` checkpoints remain blocked unless explicitly trusted.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")
    print(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="Version tag such as v0.2.7")
    parser.add_argument("--output", default="")
    parser.add_argument("--smoke", default="")
    args = parser.parse_args()
    tag = args.version if args.version.startswith("v") else f"v{args.version}"
    output = Path(args.output) if args.output else ROOT / "release_notes" / f"{tag}.md"
    smoke = Path(args.smoke) if args.smoke else ROOT / f"release-smoke-{tag}.json"
    write_notes(tag, output, smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
