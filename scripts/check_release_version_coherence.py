from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def plain(tag: str) -> str:
    return tag[1:] if tag.startswith("v") else tag


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def extract(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise AssertionError(f"Could not find {label}")
    return match.group(1)


def validate(tag: str, require_checksums: bool = True) -> None:
    tag = tag if tag.startswith("v") else f"v{tag}"
    version = plain(tag)
    setup = read("setup.bat")
    installer = read("installer/install.ps1")
    manifest = json.loads(read("installer/manifest.json"))
    checksums_path = ROOT / "installer" / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8")) if checksums_path.exists() else None
    config = json.loads(read("config.json"))
    pyproject = read("pyproject.toml")
    init = read("app/__init__.py")
    version_py = read("app/version.py")
    app_config = read("app/config.py")

    expect(extract(r'VERSION\s*=\s*"([^"]+)"', version_py, "app/version.py VERSION") == version, "app/version.py VERSION mismatch")
    expect('TAG = f"v{VERSION}"' in version_py, "app/version.py TAG mismatch")
    expect("__version__ = VERSION" in init, "app.__version__ must come from app.version.VERSION")
    expect(extract(r'"version":\s*"([^"]+)"', app_config, "app/config.py version") == version, "app/config.py version mismatch")
    expect(config.get("app", {}).get("version") == version, "config.json app.version mismatch")
    expect(extract(r'version\s*=\s*"([^"]+)"', pyproject, "pyproject version") == version, "pyproject.toml version mismatch")
    expect(extract(r"set APP_VERSION=(v[0-9]+\.[0-9]+\.[0-9]+)", setup, "setup APP_VERSION") == tag, "setup.bat APP_VERSION mismatch")
    expect(extract(r'\[string\]\$Version\s*=\s*"(v[0-9]+\.[0-9]+\.[0-9]+)"', installer, "installer Version") == tag, "installer default Version mismatch")
    expect(manifest.get("tag") == tag, "manifest tag mismatch")
    expect(manifest.get("version") == version, "manifest version mismatch")
    expect(manifest.get("app_zip") == f"Easy-ASR-Bench-{tag}-win.zip", "manifest app_zip mismatch")
    if checksums is None:
        expect(not require_checksums, "installer/checksums.json is missing")
    else:
        expect(checksums.get("version") == version, "checksums version mismatch")
        expect(f"Easy-ASR-Bench-{tag}-win.zip" in checksums.get("files", {}), "checksums ZIP name mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Release tag, for example v0.3.1")
    args = parser.parse_args()
    try:
        validate(args.tag)
    except Exception as exc:
        print(f"release version coherence failed: {exc}", file=sys.stderr)
        return 1
    print("release version coherence passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
