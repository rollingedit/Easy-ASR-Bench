from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
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


def request_json(url: str):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "Easy-ASR-Bench-release-verifier"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, destination: Path) -> None:
    headers = {"User-Agent": "Easy-ASR-Bench-release-verifier"}
    if "api.github.com" in url and "/releases/assets/" in url:
        headers["Accept"] = "application/octet-stream"
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


def asset_hashes(local_assets: dict[str, Path]) -> dict[str, str]:
    return {name: sha256(path) for name, path in sorted(local_assets.items()) if not name.startswith("release-verification-")}


def fetch_release(repo: str, tag: str) -> dict:
    api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    try:
        return request_json(api)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    releases = request_json(f"https://api.github.com/repos/{repo}/releases?per_page=100")
    if isinstance(releases, list):
        for release in releases:
            if release.get("tag_name") == tag:
                return release
    gh_release = fetch_release_with_gh(repo, tag)
    if gh_release:
        return gh_release
    raise AssertionError(f"GitHub release not found for tag {tag}")


def fetch_release_with_gh(repo: str, tag: str) -> dict | None:
    command = [
        "gh",
        "release",
        "view",
        tag,
        "--repo",
        repo,
        "--json",
        "tagName,isDraft,isPrerelease,databaseId,assets",
    ]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    data = json.loads(completed.stdout)
    assets = []
    for asset in data.get("assets", []):
        assets.append(
            {
                "name": asset.get("name"),
                "url": asset.get("apiUrl") or asset.get("url"),
                "browser_download_url": asset.get("url"),
            }
        )
    return {
        "tag_name": data.get("tagName", tag),
        "id": data.get("databaseId"),
        "draft": data.get("isDraft"),
        "prerelease": data.get("isPrerelease"),
        "assets": assets,
    }


def download_asset(asset: dict, destination: Path) -> None:
    urls = [asset.get("url"), asset.get("browser_download_url")]
    last_error: Exception | None = None
    for url in [item for item in urls if item]:
        try:
            download(url, destination)
            return
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    raise AssertionError(f"Release asset {asset.get('name', '<unknown>')} has no downloadable URL")


def download_assets(assets: dict[str, dict], destination: Path) -> dict[str, Path]:
    local_assets: dict[str, Path] = {}
    for name, asset in assets.items():
        path = destination / name
        download_asset(asset, path)
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
    if smoke.get("schema") not in {"easy_asr_bench.release_smoke.v1", "easy_asr_bench.release_smoke.v2"}:
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
    manual_rows = smoke.get("manual_rows")
    if smoke.get("schema") == "easy_asr_bench.release_smoke.v2" and not manual_rows:
        raise AssertionError(f"{smoke_name} must include explicit manual_rows, even when rows are not_run")
    manual_matrix = smoke.get("manual_matrix", {})
    if manual_matrix and not _matrix_contains_not_run(manual_matrix):
        raise AssertionError(f"{smoke_name} manual matrix does not clearly distinguish unrun rows")


def _matrix_contains_not_run(value: object) -> bool:
    if isinstance(value, dict):
        return any(_matrix_contains_not_run(child) for child in value.values())
    return value == "not_run"


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


def write_transcript(
    output: Path,
    repo: str,
    tag: str,
    expected_commit: str | None,
    resolved_commit: str,
    release: dict,
    hashes: dict[str, str],
    zip_name: str,
) -> None:
    lines = [
        "Easy ASR Bench release verification transcript",
        f"repo: {repo}",
        f"tag: {tag}",
        f"expected_commit: {expected_commit or ''}",
        f"resolved_release_commit: {resolved_commit}",
        f"release_id: {release.get('id', '')}",
        f"release_draft: {release.get('draft', '')}",
        f"release_prerelease: {release.get('prerelease', '')}",
        "",
        "downloaded_assets:",
    ]
    for name, digest in hashes.items():
        lines.append(f"  {name} {digest}")
    lines.extend(
        [
            "",
            "checks:",
            "  tag_commit_match: pass" if expected_commit else "  tag_commit_match: not_requested",
            "  required_assets_present: pass",
            "  manifest_and_checksums_valid: pass",
            "  release_smoke_asset_valid: pass",
            "  asset_hashes_match_checksums_json: pass",
            f"  zip_physical_validation: pass ({zip_name})",
            "",
            "manual_matrix_note: VM, GPU/provider, model, and media rows are not marked pass by this transcript.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def verify_release(repo: str, tag: str, expected_commit: str | None, transcript_path: Path | None = None) -> None:
    release = fetch_release(repo, tag)
    resolved_commit = resolve_tag_commit(repo, tag)
    if expected_commit and resolved_commit and not resolved_commit.startswith(expected_commit) and not expected_commit.startswith(resolved_commit):
        raise AssertionError(f"Release tag target mismatch. Expected {expected_commit}, got {resolved_commit}")

    temp = Path(tempfile.mkdtemp(prefix="easy-asr-release-"))
    try:
        assets = release_assets_by_name(release)
        local_assets = download_assets(assets, temp)
        manifest, checksums = load_release_metadata(local_assets)
        zip_name = verify_required_manifest_assets(tag, manifest, local_assets)
        verify_smoke_asset(tag, expected_commit, manifest, local_assets)
        verify_checksum_manifest(checksums, local_assets)
        validate_release_zip(local_assets[zip_name], temp)
        if transcript_path:
            write_transcript(transcript_path, repo, tag, expected_commit, resolved_commit, release, asset_hashes(local_assets), zip_name)
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--write-transcript", type=Path)
    args = parser.parse_args()
    try:
        verify_release(args.repo, args.tag, args.expected_commit, args.write_transcript)
    except Exception as exc:
        print(f"GitHub release verification failed: {exc}", file=sys.stderr)
        return 1
    print("GitHub release verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
