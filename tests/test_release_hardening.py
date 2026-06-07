import hashlib
import json
import re
from unittest.mock import patch
from pathlib import Path

import scripts.verify_github_release as verify_github_release
from scripts.check_release_version_coherence import validate as validate_version_coherence
from scripts.validate_raw_github_files import validate_bytes
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


def test_setup_discovers_supported_python_matrix():
    setup = (ROOT / "setup.bat").read_text(encoding="utf-8")

    assert "for %%V in (3.14 3.13 3.12 3.11 3.10)" in setup
    assert "winget install -e --id Python.Python.3.12" in setup
    assert "Python 3.10 through 3.14 was not found" in setup


def test_release_smoke_writer_and_verifier_require_smoke_asset():
    smoke_writer = (ROOT / "scripts" / "write_release_smoke.py").read_text(encoding="utf-8")
    verifier = (ROOT / "scripts" / "verify_github_release.py").read_text(encoding="utf-8")
    publish = (ROOT / ".github" / "workflows" / "publish-release.yml").read_text(encoding="utf-8")

    assert "easy_asr_bench.release_smoke.v1" in smoke_writer
    assert "\"not_run\"" in smoke_writer
    assert "release-smoke-$tag.json" in publish
    assert "Release smoke asset is missing" in verifier
    assert "asset_hashes_verified" in verifier


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


def test_physical_files_validate_repo_bytes():
    validate_root(ROOT)
