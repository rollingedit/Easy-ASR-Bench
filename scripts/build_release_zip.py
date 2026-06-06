from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from check_release_version_coherence import validate as validate_version_coherence


ROOT = Path(__file__).resolve().parents[1]
INSTALLER_SHA_PATTERN = r"set INSTALLER_SHA256=sha256:[0-9a-fA-F]+"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_files() -> list[str]:
    completed = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=True)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def file_bytes(rel: str, update_metadata: bool) -> bytes:
    del update_metadata
    data = (ROOT / rel).read_bytes()
    if b"\0" in data:
        return data
    suffix = Path(rel).suffix.lower()
    name = Path(rel).name.lower()
    crlf_suffixes = {".bat", ".cmd", ".ps1"}
    lf_suffixes = {".py", ".json", ".md", ".toml", ".ini", ".yml", ".yaml", ".html", ".css", ".js", ".txt"}
    if suffix in crlf_suffixes:
        text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        return text.replace("\n", "\r\n").encode("utf-8")
    if suffix in lf_suffixes or name in {".gitattributes", ".gitignore", ".editorconfig", "license"}:
        text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        return text.encode("utf-8")
    return data


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, indent=2))
        handle.write("\n")


def update_setup_installer_hash(expected_hash: str, update_metadata: bool) -> None:
    setup = ROOT / "setup.bat"
    text = setup.read_text(encoding="utf-8")
    replacement = f"set INSTALLER_SHA256=sha256:{expected_hash}"
    import re

    updated, count = re.subn(INSTALLER_SHA_PATTERN, replacement, text)
    if count != 1:
        raise SystemExit("setup.bat must contain exactly one INSTALLER_SHA256 placeholder")
    if update_metadata:
        setup.write_text(updated, encoding="utf-8", newline="\r\n")
    elif updated != text:
        raise SystemExit("setup.bat INSTALLER_SHA256 does not match installer/install.ps1")


def build(version: str, update_metadata: bool, strict_checksums: bool = False) -> Path:
    tag = version if version.startswith("v") else f"v{version}"
    plain = tag[1:]
    zip_name = f"Easy-ASR-Bench-{tag}-win.zip"
    dist = ROOT / "dist"
    stage = dist / f"Easy-ASR-Bench-{tag}"
    zip_path = dist / zip_name
    installer_hash = sha256(ROOT / "installer" / "install.ps1")
    update_setup_installer_hash(installer_hash, update_metadata)
    manifest = {
        "schema": "easy_asr_bench.installer_manifest.v2",
        "tag": tag,
        "version": plain,
        "app_zip": zip_name,
        "installer_asset": "install.ps1",
        "install_dir": "%LOCALAPPDATA%\\Easy-ASR-Bench",
        "entrypoints": ["setup.bat", "Run.bat", "Drop_Audio_Or_Folders_Here.bat", "Open_Latest_Report.bat"],
    }
    if update_metadata:
        write_json(ROOT / "installer" / "manifest.json", manifest)
    else:
        committed_manifest = json.loads((ROOT / "installer" / "manifest.json").read_text(encoding="utf-8"))
        if committed_manifest != manifest:
            print("Generated manifest:")
            print(json.dumps(manifest, indent=2))
            print("Committed manifest:")
            print(json.dumps(committed_manifest, indent=2))
            raise SystemExit("installer/manifest.json does not match generated release metadata")

    dist.mkdir(exist_ok=True)
    shutil.rmtree(stage, ignore_errors=True)
    if zip_path.exists():
        zip_path.unlink()
    stage.mkdir(parents=True)
    files = [rel for rel in git_files() if rel.replace("\\", "/") != "installer/checksums.json" and (ROOT / rel).is_file()]

    for rel in files:
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
        for rel in sorted(files):
            info = zipfile.ZipInfo((Path(f"Easy-ASR-Bench-{tag}") / rel).as_posix(), fixed_time)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 0
            info.create_version = 20
            info.extract_version = 20
            info.external_attr = 0
            info.extra = b""
            info.comment = b""
            archive.writestr(info, file_bytes(rel, update_metadata))
        archive.comment = b""

    checksums = {
        "schema": "easy_asr_bench.checksums.v1",
        "version": plain,
        "files": {
            zip_name: f"sha256:{sha256(zip_path)}",
            "setup.bat": f"sha256:{sha256(ROOT / 'setup.bat')}",
            "install.ps1": f"sha256:{installer_hash}",
            "manifest.json": f"sha256:{sha256(ROOT / 'installer' / 'manifest.json')}",
        },
    }
    if update_metadata:
        write_json(ROOT / "installer" / "checksums.json", checksums)
    else:
        committed_checksums = json.loads((ROOT / "installer" / "checksums.json").read_text(encoding="utf-8"))
        if strict_checksums and committed_checksums != checksums:
            print("Generated checksums:")
            print(json.dumps(checksums, indent=2))
            print("Committed checksums:")
            print(json.dumps(committed_checksums, indent=2))
            raise SystemExit("installer/checksums.json does not match generated release checksums")
        if committed_checksums != checksums:
            print("warning: installer/checksums.json does not match generated release checksums")
            print("warning: use --update-metadata when preparing local release metadata, or let the GitHub publish workflow upload generated checksums.json")

    validate_version_coherence(tag)

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
    parser.add_argument("--strict-checksums", action="store_true")
    args = parser.parse_args()
    build(args.version, args.update_metadata, args.strict_checksums)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
