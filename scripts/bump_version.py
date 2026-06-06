from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILES = [
    "app/__init__.py",
    "app/config.py",
    "config.json",
    "pyproject.toml",
    "setup.bat",
    "installer/install.ps1",
]


def plain(tag: str) -> str:
    return tag[1:] if tag.startswith("v") else tag


def replace_regex(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated = re.sub(pattern, replacement, text)
    if updated == text:
        raise SystemExit(f"No version replacement made in {path}")
    newline = "\r\n" if path.suffix.lower() in {".bat", ".ps1"} else "\n"
    updated = updated.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)
    path.write_text(updated, encoding="utf-8", newline="")


def update_json(path: Path, version: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["app"]["version"] = version
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def current_version() -> str:
    init_text = (ROOT / "app" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init_text)
    if not match:
        raise SystemExit("Could not find app.__version__")
    return match.group(1)


def tracked_text_files() -> list[Path]:
    import subprocess

    completed = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=True)
    paths: list[Path] = []
    for rel in completed.stdout.splitlines():
        path = ROOT / rel
        if path.is_file() and path.suffix.lower() in {".py", ".json", ".toml", ".bat", ".ps1", ".md", ".yml", ".yaml", ".txt"}:
            paths.append(path)
    return paths


def assert_old_version_removed(old: str, new: str) -> None:
    stale: list[str] = []
    for path in tracked_text_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in {"CHANGELOG.md", "installer/manifest.json", "installer/checksums.json"} or rel.startswith("release_notes/"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if old in text or f"v{old}" in text:
            stale.append(rel)
    if stale:
        raise SystemExit("Old version strings remain in tracked files: " + ", ".join(stale))
    del new


def bump(tag: str) -> None:
    version = plain(tag)
    old = current_version()
    if old == version:
        print(f"Version already set to {version}")
        return
    replace_regex(ROOT / "app" / "__init__.py", r'__version__\s*=\s*"[^"]+"', f'__version__ = "{version}"')
    replace_regex(ROOT / "app" / "config.py", r'"version":\s*"[^"]+"', f'"version": "{version}"')
    update_json(ROOT / "config.json", version)
    replace_regex(ROOT / "pyproject.toml", r'version\s*=\s*"[^"]+"', f'version = "{version}"')
    replace_regex(ROOT / "setup.bat", r"set APP_VERSION=v[0-9]+\.[0-9]+\.[0-9]+", f"set APP_VERSION=v{version}")
    replace_regex(ROOT / "installer" / "install.ps1", r'\[string\]\$Version\s*=\s*"v[0-9]+\.[0-9]+\.[0-9]+"', f'[string]$Version = "v{version}"')
    assert_old_version_removed(old, version)
    print(f"Bumped {old} -> {version}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="Version tag such as v0.2.7")
    args = parser.parse_args()
    if not re.fullmatch(r"v?\d+\.\d+\.\d+", args.version):
        raise SystemExit("Version must look like vX.Y.Z")
    bump(args.version if args.version.startswith("v") else f"v{args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
