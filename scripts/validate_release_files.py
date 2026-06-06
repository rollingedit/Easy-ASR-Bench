from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from check_release_version_coherence import validate as validate_version_coherence
from validate_physical_files import validate_root


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "dist", ".pytest_cache", ".pytest_tmp", "__pycache__"}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
            source = path.read_text(encoding="utf-8")
            ast.parse(source)
            compile(source, str(path), "exec")
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise AssertionError("PyYAML is required to validate workflow YAML") from exc
    for path in (ROOT / ".github" / "workflows").glob("*.yml"):
        yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_requirements() -> None:
    for path in (ROOT / "requirements").glob("*.txt"):
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]
        for line in lines:
            if " " in line and not line.startswith(("--", "-")):
                raise AssertionError(f"{path} contains a space-separated requirement line: {line}")


def validate_installer_safety() -> None:
    installer = ROOT / "installer" / "install.ps1"
    text = installer.read_text(encoding="utf-8")
    if "python (Join-Path" in text or "\n  python " in text or "\npython " in text:
        raise AssertionError("installer/install.ps1 must not call bare python")


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
        validate_root(ROOT)
        import app

        validate_version_coherence("v" + app.__version__, require_checksums=(ROOT / "installer" / "checksums.json").exists())
        validate_line_counts()
        validate_formats()
        validate_requirements()
        validate_installer_safety()
        validate_endings()
    except Exception as exc:
        print(f"release validation failed: {exc}", file=sys.stderr)
        return 1
    print("release file validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
