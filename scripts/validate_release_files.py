from __future__ import annotations

import ast
import json
import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "dist", ".pytest_cache", ".pytest_tmp", "__pycache__"}

MIN_LINES = {
    "setup.bat": 50,
    "Run.bat": 5,
    "Drop_Audio_Or_Folders_Here.bat": 5,
    "app/main.py": 100,
    "requirements/core.txt": 5,
    "config.json": 20,
}


def raw(path: Path) -> bytes:
    return path.read_bytes()


def physical_lines(data: bytes) -> int:
    return len(data.splitlines())


def assert_no_cr_only(path: Path) -> None:
    data = raw(path)
    normalized = data.replace(b"\r\n", b"")
    if b"\r" in normalized:
        raise AssertionError(f"{path} contains CR-only line endings")


def assert_line_ending(path: Path, expected: bytes) -> None:
    data = raw(path)
    if not data:
        return
    assert_no_cr_only(path)
    if expected == b"\r\n":
        if b"\n" in data.replace(b"\r\n", b""):
            raise AssertionError(f"{path} must use CRLF")
    elif expected == b"\n":
        if b"\r\n" in data:
            raise AssertionError(f"{path} must use LF")


def validate_line_counts() -> None:
    for rel, minimum in MIN_LINES.items():
        path = ROOT / rel
        count = physical_lines(raw(path))
        if count < minimum:
            raise AssertionError(f"{rel} has {count} physical lines, expected at least {minimum}")


def validate_formats() -> None:
    for path in ROOT.rglob("*.json"):
        if not (SKIP_DIRS & set(path.parts)):
            json.loads(path.read_text(encoding="utf-8"))
    for path in ROOT.rglob("*.py"):
        if not (SKIP_DIRS & set(path.parts)):
            py_compile.compile(str(path), doraise=True)
            ast.parse(path.read_text(encoding="utf-8"))


def validate_endings() -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or (SKIP_DIRS & set(path.parts)):
            continue
        suffix = path.suffix.lower()
        if suffix in {".bat", ".cmd", ".ps1"}:
            assert_line_ending(path, b"\r\n")
        elif suffix in {".py", ".json", ".md", ".toml", ".ini", ".yml", ".yaml", ".html", ".css", ".js", ".txt"}:
            assert_line_ending(path, b"\n")


def main() -> int:
    try:
        validate_line_counts()
        validate_formats()
        validate_endings()
    except Exception as exc:
        print(f"release validation failed: {exc}", file=sys.stderr)
        return 1
    print("release file validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
