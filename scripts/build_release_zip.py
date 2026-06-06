from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_files() -> list[str]:
    completed = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=True)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, indent=2))
        handle.write("\n")


def build(version: str, update_metadata: bool) -> Path:
    tag = version if version.startswith("v") else f"v{version}"
    plain = tag[1:]
    zip_name = f"Easy-ASR-Bench-{tag}-win.zip"
    dist = ROOT / "dist"
    stage = dist / f"Easy-ASR-Bench-{tag}"
    zip_path = dist / zip_name
    dist.mkdir(exist_ok=True)
    shutil.rmtree(stage, ignore_errors=True)
    if zip_path.exists():
        zip_path.unlink()
    stage.mkdir(parents=True)

    for rel in git_files():
        if rel.replace("\\", "/") == "installer/checksums.json":
            continue
        source = ROOT / rel
        if not source.is_file():
            continue
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    fixed_time = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(dist).as_posix(), fixed_time)
                info.compress_type = zipfile.ZIP_STORED
                archive.writestr(info, path.read_bytes())

    manifest = {
        "schema": "easy_asr_bench.installer_manifest.v1",
        "tag": tag,
        "version": plain,
        "app_zip": zip_name,
        "install_dir": "%LOCALAPPDATA%\\Easy-ASR-Bench",
        "entrypoints": ["setup.bat", "Run.bat", "Drop_Audio_Or_Folders_Here.bat"],
    }
    checksums = {
        "schema": "easy_asr_bench.checksums.v1",
        "version": plain,
        "files": {
            zip_name: f"sha256:{sha256(zip_path)}",
            "setup.bat": f"sha256:{sha256(ROOT / 'setup.bat')}",
        },
    }
    if update_metadata:
        write_json(ROOT / "installer" / "manifest.json", manifest)
        write_json(ROOT / "installer" / "checksums.json", checksums)
    else:
        committed_manifest = json.loads((ROOT / "installer" / "manifest.json").read_text(encoding="utf-8"))
        committed_checksums = json.loads((ROOT / "installer" / "checksums.json").read_text(encoding="utf-8"))
        if committed_manifest != manifest:
            raise SystemExit("installer/manifest.json does not match generated release metadata")
        if committed_checksums != checksums:
            raise SystemExit("installer/checksums.json does not match generated release checksums")

    verify_dir = dist / f"verify-{tag}"
    shutil.rmtree(verify_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(verify_dir)
    validator = verify_dir / f"Easy-ASR-Bench-{tag}" / "scripts" / "validate_release_files.py"
    subprocess.run([sys.executable, str(validator)], cwd=validator.parents[1], check=True)
    print(zip_path)
    print(f"zip sha256: {sha256(zip_path)}")
    print(f"setup.bat sha256: {sha256(ROOT / 'setup.bat')}")
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--update-metadata", action="store_true")
    args = parser.parse_args()
    build(args.version, args.update_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
