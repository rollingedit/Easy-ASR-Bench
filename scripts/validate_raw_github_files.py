from __future__ import annotations

import argparse
import sys
import urllib.request
from dataclasses import dataclass


DEFAULT_EXPECTATIONS = {
    "setup.bat": 200,
    "installer/install.ps1": 250,
    "scripts/validate_physical_files.py": 150,
    "scripts/verify_github_release.py": 150,
    ".github/workflows/release-gate.yml": 75,
    ".github/workflows/publish-release.yml": 50,
    "app/model_scanner.py": 600,
    "app/results_writer.py": 100,
    "app/scoring.py": 80,
}


@dataclass(frozen=True)
class Expectation:
    path: str
    minimum_lines: int


def parse_expectation(value: str) -> Expectation:
    try:
        path, minimum = value.rsplit(":", 1)
        return Expectation(path=path, minimum_lines=int(minimum))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--expect values must look like path:min_lines") from exc


def raw_url(repo: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Easy-ASR-Bench-raw-validator"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def physical_line_count(data: bytes) -> int:
    if not data:
        return 0
    return len(data.splitlines())


def validate_bytes(path: str, data: bytes, minimum_lines: int) -> None:
    if b"\r" in data.replace(b"\r\n", b""):
        raise AssertionError(f"{path} contains CR-only line endings in raw GitHub bytes")
    line_count = physical_line_count(data)
    if line_count < minimum_lines:
        raise AssertionError(f"{path} has {line_count} raw physical lines, expected at least {minimum_lines}")


def validate(repo: str, commit: str, expectations: list[Expectation]) -> None:
    for expectation in expectations:
        url = raw_url(repo, commit, expectation.path)
        validate_bytes(expectation.path, fetch(url), expectation.minimum_lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--commit", required=True, help="Commit SHA to fetch from raw.githubusercontent.com")
    parser.add_argument("--expect", action="append", type=parse_expectation, help="path:min_lines; may be repeated")
    args = parser.parse_args()
    expectations = args.expect or [Expectation(path, lines) for path, lines in DEFAULT_EXPECTATIONS.items()]
    try:
        validate(args.repo, args.commit, expectations)
    except Exception as exc:
        print(f"raw GitHub file validation failed: {exc}", file=sys.stderr)
        return 1
    print("raw GitHub file validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
