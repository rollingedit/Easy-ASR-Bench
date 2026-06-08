from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXPECTATIONS = {
    "setup.bat": "any",
    "installer/install.ps1": "any",
    "scripts/validate_physical_files.py": "lf",
    "scripts/verify_github_release.py": "lf",
    ".github/workflows/release-gate.yml": "lf",
    ".github/workflows/publish-release.yml": "lf",
    "app/model_scanner.py": "lf",
    "app/results_writer.py": "lf",
    "app/scoring.py": "lf",
}


@dataclass(frozen=True)
class Expectation:
    path: str
    minimum_lines: int | None = None
    line_ending: str = "any"


@dataclass(frozen=True)
class ByteDiagnostics:
    path: str
    byte_count: int
    crlf_count: int
    lf_count: int
    bare_cr_count: int
    bare_lf_count: int
    physical_line_count_crlf: int
    physical_line_count_universal: int
    first_32_bytes_hex: str
    last_32_bytes_hex: str


def parse_expectation(value: str) -> Expectation:
    try:
        parts = value.rsplit(":", 2)
        if len(parts) == 1:
            path = parts[0]
            minimum = None
            line_ending = "any"
        elif len(parts) == 2:
            path, second = parts
            if second.lower() in {"any", "crlf", "lf"}:
                minimum = None
                line_ending = second
            else:
                minimum = int(second)
                line_ending = "any"
        else:
            path, minimum, line_ending = parts
            minimum = int(minimum)
        line_ending = line_ending.lower()
        if line_ending not in {"any", "crlf", "lf"}:
            raise ValueError
        return Expectation(path=path, minimum_lines=minimum, line_ending=line_ending)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--expect values must look like path[:min_lines][:any|crlf|lf]") from exc


def raw_url(repo: str, ref: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Easy-ASR-Bench-raw-validator"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def physical_line_count(data: bytes) -> int:
    if not data:
        return 0
    return len(data.splitlines())


def byte_diagnostics(path: str, data: bytes) -> ByteDiagnostics:
    crlf_count = data.count(b"\r\n")
    lf_count = data.count(b"\n")
    bare_cr_count = data.replace(b"\r\n", b"").count(b"\r")
    bare_lf_count = lf_count - crlf_count
    return ByteDiagnostics(
        path=path,
        byte_count=len(data),
        crlf_count=crlf_count,
        lf_count=lf_count,
        bare_cr_count=bare_cr_count,
        bare_lf_count=bare_lf_count,
        physical_line_count_crlf=crlf_count + (1 if data and not data.endswith(b"\r\n") else 0),
        physical_line_count_universal=physical_line_count(data),
        first_32_bytes_hex=data[:32].hex(),
        last_32_bytes_hex=data[-32:].hex(),
    )


def format_diagnostics(diagnostics: ByteDiagnostics) -> str:
    return "\n".join(
        [
            f"file: {diagnostics.path}",
            f"  byte_count: {diagnostics.byte_count}",
            f"  count_\\r\\n: {diagnostics.crlf_count}",
            f"  count_\\n: {diagnostics.lf_count}",
            f"  count_bare_\\r: {diagnostics.bare_cr_count}",
            f"  count_bare_\\n: {diagnostics.bare_lf_count}",
            f"  crlf_terminated_line_count_diagnostic: {diagnostics.physical_line_count_crlf}",
            f"  physical_line_count_universal: {diagnostics.physical_line_count_universal}",
            f"  first_32_bytes_hex: {diagnostics.first_32_bytes_hex}",
            f"  last_32_bytes_hex: {diagnostics.last_32_bytes_hex}",
            "  note: raw GitHub serves canonical Git blob bytes; line counts are diagnostics, not behavior proof.",
        ]
    )


def validate_bytes(path: str, data: bytes, minimum_lines: int | None = None, line_ending: str = "any") -> ByteDiagnostics:
    diagnostics = byte_diagnostics(path, data)
    if diagnostics.byte_count == 0:
        raise AssertionError(f"{path} is empty in raw GitHub bytes")
    if b"\r" in data.replace(b"\r\n", b""):
        raise AssertionError(f"{path} contains CR-only line endings in raw GitHub bytes")
    if line_ending == "crlf" and diagnostics.bare_lf_count:
        raise AssertionError(f"{path} must use CRLF line endings in raw GitHub bytes")
    if line_ending == "lf" and diagnostics.crlf_count:
        raise AssertionError(f"{path} must use LF line endings in raw GitHub bytes")
    if minimum_lines is not None and diagnostics.physical_line_count_universal < minimum_lines:
        raise AssertionError(
            f"{path} has {diagnostics.physical_line_count_universal} raw physical lines, expected at least {minimum_lines}"
        )
    return diagnostics


def validate(repo: str, ref: str, expectations: list[Expectation]) -> list[ByteDiagnostics]:
    diagnostics: list[ByteDiagnostics] = []
    for expectation in expectations:
        url = raw_url(repo, ref, expectation.path)
        diagnostics.append(validate_bytes(expectation.path, fetch(url), expectation.minimum_lines, expectation.line_ending))
    return diagnostics


def compare_raw_to_zip(raw_files: dict[str, bytes], zip_path: Path) -> None:
    temp = Path(tempfile.mkdtemp(prefix="easy-asr-raw-zip-"))
    try:
        shutil.unpack_archive(str(zip_path), str(temp), "zip")
        roots = [path for path in temp.iterdir() if path.is_dir()]
        app_root = roots[0] if len(roots) == 1 else temp
        for path, raw_data in raw_files.items():
            zip_file = app_root / path
            if not zip_file.exists():
                raise AssertionError(f"{path} is missing from release ZIP copy")
            zip_data = zip_file.read_bytes()
            if raw_data.replace(b"\r\n", b"\n") != zip_data.replace(b"\r\n", b"\n"):
                raise AssertionError(f"{path} raw GitHub content differs from release ZIP copy")
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def validate_with_data(repo: str, ref: str, expectations: list[Expectation]) -> tuple[list[ByteDiagnostics], dict[str, bytes]]:
    diagnostics: list[ByteDiagnostics] = []
    raw_files: dict[str, bytes] = {}
    for expectation in expectations:
        url = raw_url(repo, ref, expectation.path)
        data = fetch(url)
        raw_files[expectation.path] = data
        diagnostics.append(validate_bytes(expectation.path, data, expectation.minimum_lines, expectation.line_ending))
    return diagnostics, raw_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--commit", help="Commit SHA to fetch from raw.githubusercontent.com")
    parser.add_argument("--ref", help="Tag, branch, or commit to fetch from raw.githubusercontent.com")
    parser.add_argument("--expect", action="append", type=parse_expectation, help="path[:min_lines][:any|crlf|lf]; may be repeated")
    parser.add_argument("--zip", type=Path, help="Optional release ZIP whose matching source files must byte-match raw GitHub")
    args = parser.parse_args()
    ref = args.ref or args.commit
    if not ref:
        parser.error("--ref or --commit is required")
    expectations = args.expect or [
        Expectation(path, None, line_ending)
        for path, line_ending in DEFAULT_EXPECTATIONS.items()
    ]
    try:
        diagnostics, raw_files = validate_with_data(args.repo, ref, expectations)
        if args.zip:
            compare_raw_to_zip(raw_files, args.zip)
    except Exception as exc:
        print(f"raw GitHub file validation failed: {exc}", file=sys.stderr)
        return 1
    for item in diagnostics:
        print(format_diagnostics(item))
    print("raw GitHub file validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
