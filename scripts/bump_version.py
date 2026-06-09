from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILES = [
    "app/version.py",
    "app/__init__.py",
    "app/config.py",
    "config.json",
    "pyproject.toml",
    "setup.bat",
    "installer/install.ps1",
    "installer/manifest.json",
    "installer/checksums.json",
    "qa/runtime_matrix/rows/clean_vm_bootstrap.py",
    "qa/runtime_matrix/rows/installer_validation.py",
    "docs/release_verification.md",
]


def plain(tag: str) -> str:
    return tag[1:] if tag.startswith("v") else tag


def replace_regex(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text)
    if count == 0:
        raise SystemExit(f"No version replacement made in {path}")
    newline = "\r\n" if path.suffix.lower() in {".bat", ".ps1"} else "\n"
    updated = updated.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)
    path.write_text(updated, encoding="utf-8", newline="")


def update_json(path: Path, version: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["app"]["version"] = version
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def update_installer_manifest(path: Path, version: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tag"] = f"v{version}"
    data["version"] = version
    data["app_zip"] = f"Easy-ASR-Bench-v{version}-win.zip"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def update_checksums(path: Path, old: str, version: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["version"] = version
    old_zip = f"Easy-ASR-Bench-v{old}-win.zip"
    new_zip = f"Easy-ASR-Bench-v{version}-win.zip"
    files = data.get("files", {})
    if old_zip in files and new_zip not in files:
        files[new_zip] = files.pop(old_zip)
    for name in list(files):
        if re.fullmatch(r"Easy-ASR-Bench-v\d+\.\d+\.\d+-win\.zip", name) and name != new_zip:
            files[new_zip] = files.pop(name)
    data["files"] = files
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def current_version() -> str:
    version_text = (ROOT / "app" / "version.py").read_text(encoding="utf-8")
    match = re.search(r'VERSION\s*=\s*"([^"]+)"', version_text)
    if not match:
        raise SystemExit("Could not find app.version.VERSION")
    return match.group(1)


def tracked_text_files() -> list[Path]:
    release_inputs = [
        "app/version.py",
        "app/__init__.py",
        "app/config.py",
        "config.json",
        "pyproject.toml",
        "setup.bat",
        "installer/install.ps1",
        "installer/manifest.json",
        "installer/checksums.json",
        "README.md",
        "docs/supported_models.md",
        "docs/release_verification.md",
        "docs/troubleshooting.md",
        "qa/runtime_matrix/rows/clean_vm_bootstrap.py",
        "qa/runtime_matrix/rows/installer_validation.py",
    ]
    paths: list[Path] = []
    for rel in release_inputs:
        path = ROOT / rel
        if path.is_file():
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
    replace_regex(ROOT / "app" / "version.py", r'VERSION\s*=\s*"[^"]+"', f'VERSION = "{version}"')
    replace_regex(ROOT / "app" / "config.py", r'"version":\s*"[^"]+"', f'"version": "{version}"')
    update_json(ROOT / "config.json", version)
    replace_regex(ROOT / "pyproject.toml", r'version\s*=\s*"[^"]+"', f'version = "{version}"')
    replace_regex(ROOT / "setup.bat", r"set APP_VERSION=v[0-9]+\.[0-9]+\.[0-9]+", f"set APP_VERSION=v{version}")
    replace_regex(ROOT / "installer" / "install.ps1", r'\[string\]\$Version\s*=\s*"v[0-9]+\.[0-9]+\.[0-9]+"', f'[string]$Version = "v{version}"')
    update_installer_manifest(ROOT / "installer" / "manifest.json", version)
    update_checksums(ROOT / "installer" / "checksums.json", old, version)
    replace_regex(ROOT / "qa" / "runtime_matrix" / "rows" / "clean_vm_bootstrap.py", r"v[0-9]+\.[0-9]+\.[0-9]-sandbox", f"v{version}-sandbox")
    replace_regex(ROOT / "qa" / "runtime_matrix" / "rows" / "clean_vm_bootstrap.py", r"--tag v[0-9]+\.[0-9]+\.[0-9]", f"--tag v{version}")
    replace_regex(ROOT / "qa" / "runtime_matrix" / "rows" / "installer_validation.py", r'VERSION = "v[0-9]+\.[0-9]+\.[0-9]+"', f'VERSION = "v{version}"')
    replace_regex(ROOT / "docs" / "release_verification.md", r"v[0-9]+\.[0-9]+\.[0-9]", f"v{version}")
    if old != version:
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
