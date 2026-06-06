from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

try:
    from validate_physical_files import validate_root
except ModuleNotFoundError:
    from scripts.validate_physical_files import validate_root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "Easy-ASR-Bench-release-verifier"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Easy-ASR-Bench-release-verifier"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def resolve_tag_commit(repo: str, tag: str) -> str:
    ref = request_json(f"https://api.github.com/repos/{repo}/git/ref/tags/{tag}")
    target = ref.get("object", {})
    target_sha = target.get("sha", "")
    if target.get("type") == "tag" and target_sha:
        tag_object = request_json(f"https://api.github.com/repos/{repo}/git/tags/{target_sha}")
        target_sha = tag_object.get("object", {}).get("sha", target_sha)
    return target_sha


def verify_release(repo: str, tag: str, expected_commit: str | None) -> None:
    api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    release = request_json(api)
    if expected_commit:
        target_sha = resolve_tag_commit(repo, tag)
        if target_sha and not target_sha.startswith(expected_commit) and not expected_commit.startswith(target_sha):
            raise AssertionError(f"Release tag target mismatch. Expected {expected_commit}, got {target_sha}")

    temp = Path(tempfile.mkdtemp(prefix="easy-asr-release-"))
    try:
        assets = {asset["name"]: asset for asset in release.get("assets", [])}
        required = {"setup.bat", "install.ps1", "manifest.json", "checksums.json"}
        missing = sorted(required - set(assets))
        if missing:
            raise AssertionError("Missing release assets: " + ", ".join(missing))

        local_assets: dict[str, Path] = {}
        for name, asset in assets.items():
            path = temp / name
            download(asset["browser_download_url"], path)
            local_assets[name] = path

        manifest = json.loads(local_assets["manifest.json"].read_text(encoding="utf-8"))
        checksums = json.loads(local_assets["checksums.json"].read_text(encoding="utf-8"))
        if manifest.get("schema") == "easy_asr_bench.installer_manifest.v2" and manifest.get("installer_asset") != "install.ps1":
            raise AssertionError("v2 manifest must declare installer_asset as install.ps1")
        zip_name = manifest["app_zip"]
        if zip_name not in local_assets:
            raise AssertionError(f"Release ZIP asset is missing: {zip_name}")

        for name, expected_hash in checksums.get("files", {}).items():
            if name not in local_assets:
                raise AssertionError(f"checksums.json names an asset that was not uploaded: {name}")
            actual = sha256(local_assets[name])
            if actual != expected_hash:
                raise AssertionError(f"Checksum mismatch for {name}. Expected {expected_hash}, got {actual}")

        extract_dir = temp / "extract"
        shutil.unpack_archive(str(local_assets[zip_name]), str(extract_dir), "zip")
        roots = [path for path in extract_dir.iterdir() if path.is_dir()]
        app_root = roots[0] if len(roots) == 1 else extract_dir
        validate_root(app_root)
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    try:
        verify_release(args.repo, args.tag, args.expected_commit)
    except Exception as exc:
        print(f"GitHub release verification failed: {exc}", file=sys.stderr)
        return 1
    print("GitHub release verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
