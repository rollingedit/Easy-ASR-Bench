from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "Easy-ASR-Bench-release-verifier"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, destination: Path) -> None:
    headers = {"User-Agent": "Easy-ASR-Bench-release-verifier"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
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


def assert_commit_matches(repo: str, tag: str, expected_commit: str | None) -> None:
    if not expected_commit:
        return
    target_sha = resolve_tag_commit(repo, tag)
    if target_sha and not target_sha.startswith(expected_commit) and not expected_commit.startswith(target_sha):
        raise AssertionError(f"Release tag target mismatch. Expected {expected_commit}, got {target_sha}")


def release_assets_by_name(release: dict) -> dict[str, dict]:
    assets = {asset["name"]: asset for asset in release.get("assets", [])}
    required = {"setup.bat", "install.ps1", "manifest.json", "checksums.json"}
    missing = sorted(required - set(assets))
    if missing:
        raise AssertionError("Missing release assets: " + ", ".join(missing))
    return assets


def download_assets(assets: dict[str, dict], destination: Path) -> dict[str, Path]:
    local_assets: dict[str, Path] = {}
    for name, asset in assets.items():
        path = destination / name
        download(asset["browser_download_url"], path)
        local_assets[name] = path
    return local_assets


def load_release_metadata(local_assets: dict[str, Path]) -> tuple[dict, dict]:
    manifest = json.loads(local_assets["manifest.json"].read_text(encoding="utf-8"))
    checksums = json.loads(local_assets["checksums.json"].read_text(encoding="utf-8"))
    if manifest.get("schema") == "easy_asr_bench.installer_manifest.v2" and manifest.get("installer_asset") != "install.ps1":
        raise AssertionError("v2 manifest must declare installer_asset as install.ps1")
    return manifest, checksums


def verify_required_manifest_assets(tag: str, manifest: dict, local_assets: dict[str, Path]) -> str:
    if manifest.get("tag") and manifest["tag"] != tag:
        raise AssertionError(f"manifest tag mismatch. Expected {tag}, got {manifest['tag']}")
    zip_name = manifest["app_zip"]
    if zip_name not in local_assets:
        raise AssertionError(f"Release ZIP asset is missing: {zip_name}")
    if manifest.get("schema") == "easy_asr_bench.installer_manifest.v2":
        smoke_name = f"release-smoke-{tag}.json"
        if smoke_name not in local_assets:
            raise AssertionError(f"Release smoke asset is missing: {smoke_name}")
    return zip_name


def verify_smoke_asset(tag: str, expected_commit: str | None, manifest: dict, local_assets: dict[str, Path]) -> None:
    if manifest.get("schema") != "easy_asr_bench.installer_manifest.v2":
        return
    smoke_name = f"release-smoke-{tag}.json"
    smoke = json.loads(local_assets[smoke_name].read_text(encoding="utf-8"))
    if smoke.get("schema") != "easy_asr_bench.release_smoke.v1":
        raise AssertionError(f"{smoke_name} has an unexpected schema")
    if smoke.get("tag") != tag:
        raise AssertionError(f"{smoke_name} tag mismatch")
    if expected_commit and smoke.get("commit") and not smoke["commit"].startswith(expected_commit) and not expected_commit.startswith(smoke["commit"]):
        raise AssertionError(f"{smoke_name} commit mismatch")
    if smoke.get("asset_hashes_verified") is not True:
        raise AssertionError(f"{smoke_name} does not record verified asset hashes")
    checks = smoke.get("checks", [])
    failed = [check.get("name", "unknown") for check in checks if check.get("status") == "fail"]
    if failed:
        raise AssertionError(f"{smoke_name} records failed automated checks: {', '.join(failed)}")
    manual_matrix = smoke.get("manual_matrix", {})
    if manual_matrix and not any(value == "not_run" or isinstance(value, dict) for value in manual_matrix.values()):
        raise AssertionError(f"{smoke_name} manual matrix does not clearly distinguish unrun rows")


def verify_checksum_manifest(checksums: dict, local_assets: dict[str, Path]) -> None:
    files = checksums.get("files", {})
    if not files:
        raise AssertionError("checksums.json does not list any release files")
    for name, expected_hash in files.items():
        if name not in local_assets:
            raise AssertionError(f"checksums.json names an asset that was not uploaded: {name}")
        actual = sha256(local_assets[name])
        if actual != expected_hash:
            raise AssertionError(f"Checksum mismatch for {name}. Expected {expected_hash}, got {actual}")


def validate_release_zip(zip_path: Path, temp: Path) -> None:
    extract_dir = temp / "extract"
    shutil.unpack_archive(str(zip_path), str(extract_dir), "zip")
    roots = [path for path in extract_dir.iterdir() if path.is_dir()]
    app_root = roots[0] if len(roots) == 1 else extract_dir
    validate_root(app_root)


def verify_release(repo: str, tag: str, expected_commit: str | None) -> None:
    api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    release = request_json(api)
    assert_commit_matches(repo, tag, expected_commit)

    temp = Path(tempfile.mkdtemp(prefix="easy-asr-release-"))
    try:
        assets = release_assets_by_name(release)
        local_assets = download_assets(assets, temp)
        manifest, checksums = load_release_metadata(local_assets)
        zip_name = verify_required_manifest_assets(tag, manifest, local_assets)
        verify_smoke_asset(tag, expected_commit, manifest, local_assets)
        verify_checksum_manifest(checksums, local_assets)
        validate_release_zip(local_assets[zip_name], temp)
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
