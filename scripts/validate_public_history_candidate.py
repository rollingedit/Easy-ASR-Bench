from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    from validate_public_hygiene import scan_history, scan_paths, tracked_files
except ModuleNotFoundError:
    from scripts.validate_public_hygiene import scan_history, scan_paths, tracked_files


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, check=True)


def tree_entries(repo: Path, ref: str) -> dict[str, tuple[str, str, str]]:
    completed = _run(["git", "ls-tree", "-r", "-z", ref], repo)
    entries: dict[str, tuple[str, str, str]] = {}
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        meta, path = raw.split(b"\t", 1)
        mode, kind, sha = meta.decode("ascii").split(" ")
        entries[path.decode("utf-8", errors="surrogateescape")] = (mode, kind, sha)
    return entries


def blob_bytes(repo: Path, ref: str, path: str) -> bytes:
    return _run(["git", "show", f"{ref}:{path}"], repo).stdout


def compare_trees(source_repo: Path, source_ref: str, candidate_repo: Path, candidate_ref: str) -> list[str]:
    errors: list[str] = []
    source_entries = tree_entries(source_repo, source_ref)
    candidate_entries = tree_entries(candidate_repo, candidate_ref)
    source_paths = set(source_entries)
    candidate_paths = set(candidate_entries)
    for path in sorted(source_paths - candidate_paths):
        errors.append(f"candidate is missing path: {path}")
    for path in sorted(candidate_paths - source_paths):
        errors.append(f"candidate has extra path: {path}")
    for path in sorted(source_paths & candidate_paths):
        source_mode, source_kind, _source_sha = source_entries[path]
        candidate_mode, candidate_kind, _candidate_sha = candidate_entries[path]
        if (source_mode, source_kind) != (candidate_mode, candidate_kind):
            errors.append(f"candidate mode/type differs for {path}: {candidate_mode} {candidate_kind}, expected {source_mode} {source_kind}")
            continue
        if source_kind == "blob" and blob_bytes(source_repo, source_ref, path) != blob_bytes(candidate_repo, candidate_ref, path):
            errors.append(f"candidate blob content differs for {path}")
    return errors


def validate_candidate(source_repo: Path, source_ref: str, candidate_repo: Path, candidate_ref: str) -> list[str]:
    errors = compare_trees(source_repo, source_ref, candidate_repo, candidate_ref)
    errors.extend("candidate tree hygiene: " + error for error in scan_paths(tracked_files(candidate_repo)))
    errors.extend("candidate history hygiene: " + error for error in scan_history([candidate_ref], root=candidate_repo, max_findings=50))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a sanitized public-history candidate repo before branch replacement.")
    parser.add_argument("--source-repo", default=ROOT, type=Path)
    parser.add_argument("--source-ref", default="main")
    parser.add_argument("--candidate-repo", required=True, type=Path)
    parser.add_argument("--candidate-ref", default="main")
    args = parser.parse_args()
    errors = validate_candidate(
        args.source_repo.resolve(),
        args.source_ref,
        args.candidate_repo.resolve(),
        args.candidate_ref,
    )
    if errors:
        print("public history candidate validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("public history candidate validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
