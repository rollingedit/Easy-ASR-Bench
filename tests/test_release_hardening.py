import hashlib
import json
import re
import zipfile
from unittest.mock import patch
from pathlib import Path

import scripts.verify_github_release as verify_github_release
from scripts.check_release_version_coherence import validate as validate_version_coherence
from scripts.validate_raw_github_files import byte_diagnostics, compare_raw_to_zip, format_diagnostics, validate_bytes
from scripts.validate_physical_files import validate_root


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_setup_embeds_current_installer_hash():
    setup = (ROOT / "setup.bat").read_text(encoding="utf-8")
    match = re.search(r"set INSTALLER_SHA256=sha256:([0-9a-fA-F]+)", setup)

    assert match is not None
    assert match.group(1).lower() == sha256(ROOT / "installer" / "install.ps1")


def test_release_checksums_include_bootstrap_assets():
    checksums = json.loads((ROOT / "installer" / "checksums.json").read_text(encoding="utf-8"))

    assert "setup.bat" in checksums["files"]
    assert "install.ps1" in checksums["files"]
    assert "manifest.json" in checksums["files"]


def test_release_version_coherence_matches_app_version():
    import app

    validate_version_coherence("v" + app.__version__)


def test_raw_github_validator_catches_collapsed_lines():
    with patch("scripts.validate_raw_github_files.fetch", return_value=b"first\rsecond\rthird"):
        try:
            validate_bytes("setup.bat", b"first\rsecond\rthird", 200)
        except AssertionError as exc:
            assert "CR-only line endings" in str(exc)
        else:
            raise AssertionError("collapsed raw lines were accepted")


def test_raw_github_validator_reports_byte_diagnostics():
    diagnostics = byte_diagnostics("setup.bat", b"@echo off\r\necho ok\r\n")
    formatted = format_diagnostics(diagnostics)

    assert diagnostics.byte_count == 20
    assert diagnostics.crlf_count == 2
    assert diagnostics.bare_cr_count == 0
    assert diagnostics.physical_line_count_universal == 2
    assert "first_32_bytes_hex" in formatted
    assert "last_32_bytes_hex" in formatted


def test_raw_github_validator_compares_raw_bytes_to_zip_copy(tmp_path):
    zip_path = tmp_path / "release.zip"
    app_root = tmp_path / "Easy-ASR-Bench-v1"
    app_root.mkdir()
    (app_root / "setup.bat").write_bytes(b"@echo off\r\necho ok\r\n")
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(app_root / "setup.bat", "Easy-ASR-Bench-v1/setup.bat")

    compare_raw_to_zip({"setup.bat": b"@echo off\r\necho ok\r\n"}, zip_path)
    compare_raw_to_zip({"setup.bat": b"@echo off\necho ok\n"}, zip_path)

    try:
        compare_raw_to_zip({"setup.bat": b"@echo off\r\necho changed\r\n"}, zip_path)
    except AssertionError as exc:
        assert "content differs from release ZIP copy" in str(exc)
    else:
        raise AssertionError("raw/ZIP byte mismatch was accepted")


def test_installer_verify_release_checks_uploaded_bootstrap_assets():
    installer = (ROOT / "installer" / "install.ps1").read_text(encoding="utf-8")

    assert "$manifestJson.installer_asset -ne \"install.ps1\"" in installer
    assert "Assert-Checksum $Manifest $checksumsJson.files.'manifest.json' \"manifest.json\"" in installer
    assert "Assert-Checksum $releaseSetup $checksumsJson.files.'setup.bat' \"setup.bat\"" in installer
    assert "Assert-Checksum $releaseInstaller $checksumsJson.files.'install.ps1' \"install.ps1\"" in installer
    assert "Legacy manifest does not declare installer_asset" in installer
    assert "Assert-StagingPhysicalFiles" in installer
    assert "Python 3.10-3.14" in installer
    assert "Validating installed app after local setup" in installer
    assert "Installed app validation failed" in installer
    assert "[string]$AssetDir" in installer
    assert "Copy-ReleaseAsset" in installer


def test_setup_exposes_staged_asset_dir_for_prepublish_verification():
    setup = (ROOT / "setup.bat").read_text(encoding="utf-8")

    assert "--asset-dir" in setup
    assert "-AssetDir \"%ASSET_DIR%\"" in setup


def test_setup_discovers_supported_python_matrix():
    setup = (ROOT / "setup.bat").read_text(encoding="utf-8")

    assert "for %%V in (3.14 3.13 3.12 3.11 3.10)" in setup
    assert "winget install -e --id Python.Python.3.12" in setup
    assert "Python 3.10 through 3.14 was not found" in setup


def test_release_smoke_writer_and_verifier_require_smoke_asset():
    smoke_writer = (ROOT / "scripts" / "write_release_smoke.py").read_text(encoding="utf-8")
    verifier = (ROOT / "scripts" / "verify_github_release.py").read_text(encoding="utf-8")
    publish = (ROOT / ".github" / "workflows" / "publish-release.yml").read_text(encoding="utf-8")

    assert "easy_asr_bench.release_smoke.v2" in smoke_writer
    assert "\"not_run\"" in smoke_writer
    assert "manual_rows" in smoke_writer
    assert "release-smoke-$tag.json" in publish
    assert "--require-all-pass --require-log-hashes --require-environment-summary" in publish
    assert "Release smoke asset is missing" in verifier
    assert "asset_hashes_verified" in verifier
    assert "--write-transcript" in verifier


def test_publish_workflow_refuses_public_asset_mutation_before_clobber():
    publish = (ROOT / ".github" / "workflows" / "publish-release.yml").read_text(encoding="utf-8")

    assert "allow_replace_public" in publish
    assert "Refusing to replace assets on an already-public release" in publish
    assert publish.index("Refusing to replace assets on an already-public release") < publish.index("gh release upload")
    assert "setup.bat --dry-run --verify-release --asset-dir" in publish
    assert "release-verification-${{ inputs.tag }}.txt" in publish
    assert "gh release download ${{ inputs.tag }} --pattern release-verification-${{ inputs.tag }}.txt" in publish
    assert "Uploaded release verification transcript is missing required proof fields" in publish

    release_gate = (ROOT / ".github" / "workflows" / "release-gate.yml").read_text(encoding="utf-8")
    assert "validate_raw_github_files.py" in release_gate
    assert "--zip dist/Easy-ASR-Bench-$version-win.zip" in release_gate


def test_release_verifier_peels_annotated_tags():
    responses = {
        "https://api.github.com/repos/owner/repo/git/ref/tags/v1": {
            "object": {"type": "tag", "sha": "tag-object-sha"},
        },
        "https://api.github.com/repos/owner/repo/git/tags/tag-object-sha": {
            "object": {"type": "commit", "sha": "commit-sha"},
        },
    }

    with patch.object(verify_github_release, "request_json", side_effect=lambda url: responses[url]):
        assert verify_github_release.resolve_tag_commit("owner/repo", "v1") == "commit-sha"


def test_release_verifier_finds_draft_release_when_tag_endpoint_404():
    def fake_request_json(url):
        if url.endswith("/releases/tags/v1"):
            raise verify_github_release.urllib.error.HTTPError(url, 404, "Not Found", None, None)
        if url.endswith("/releases?per_page=100"):
            return [{"tag_name": "v1", "assets": []}]
        raise AssertionError(url)

    with patch.object(verify_github_release, "request_json", side_effect=fake_request_json):
        assert verify_github_release.fetch_release("owner/repo", "v1")["tag_name"] == "v1"


def test_release_verifier_falls_back_to_gh_for_draft_release():
    def fake_run(command, text, capture_output, check):
        class Completed:
            returncode = 0
            stdout = json.dumps(
                {
                    "tagName": "v1",
                    "isDraft": True,
                    "isPrerelease": False,
                    "databaseId": 123,
                    "assets": [
                        {
                            "name": "setup.bat",
                            "apiUrl": "https://api.github.com/repos/owner/repo/releases/assets/1",
                            "url": "https://github.com/owner/repo/releases/download/untagged/setup.bat",
                        }
                    ],
                }
            )

        assert command[:4] == ["gh", "release", "view", "v1"]
        return Completed()

    with patch.object(verify_github_release.subprocess, "run", side_effect=fake_run):
        release = verify_github_release.fetch_release_with_gh("owner/repo", "v1")

    assert release is not None
    assert release["tag_name"] == "v1"
    assert release["draft"] is True
    assert release["assets"][0]["url"].endswith("/releases/assets/1")


def test_physical_files_validate_repo_bytes():
    validate_root(ROOT)
