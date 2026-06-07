import json
from pathlib import Path

from scripts.validate_release_smoke import validate_smoke
from scripts.verify_release_transcript import verify_transcript
from scripts.write_release_verification_manifest import build_manifest


def test_validate_release_smoke_fails_missing_required_row():
    errors = validate_smoke({"schema": "easy_asr_bench.release_smoke.v2", "manual_rows": []}, ["win11_clean_no_python_setup"])

    assert any("missing required smoke row" in error for error in errors)


def test_validate_release_smoke_requires_pass_when_strict():
    smoke = {"schema": "easy_asr_bench.release_smoke.v2", "manual_rows": [{"id": "win11_clean_no_python_setup", "status": "not_run"}]}

    errors = validate_smoke(smoke, ["win11_clean_no_python_setup"], require_all_pass=True)

    assert "win11_clean_no_python_setup status is not_run, expected pass" in errors


def test_validate_release_smoke_requires_version_and_commit_when_strict():
    smoke = {"schema": "easy_asr_bench.release_smoke.v2", "manual_rows": [{"id": "win11_clean_no_python_setup", "status": "pass"}]}

    errors = validate_smoke(smoke, ["win11_clean_no_python_setup"], require_all_pass=True)

    assert "win11_clean_no_python_setup is missing app_version" in errors
    assert "win11_clean_no_python_setup is missing release_commit" in errors


def test_validate_release_smoke_requires_evidence_fields_when_strict():
    smoke = {"schema": "easy_asr_bench.release_smoke.v2", "manual_rows": [{"id": "win11_clean_no_python_setup", "status": "pass"}]}

    errors = validate_smoke(smoke, ["win11_clean_no_python_setup"], require_log_hashes=True, require_environment_summary=True)

    assert any("logs_sha256" in error for error in errors)
    assert any("environment_summary" in error for error in errors)


def test_verify_release_transcript_rejects_self_hash_and_checksum_mismatch(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "setup.bat").write_bytes(b"setup")
    (assets / "release-verification-v1.txt").write_text(
        "Easy ASR Bench release verification transcript\n"
        "downloaded_assets:\n"
        "  setup.bat sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "  release-verification-v1.txt sha256:1111111111111111111111111111111111111111111111111111111111111111\n",
        encoding="utf-8",
    )
    (assets / "checksums.json").write_text(
        json.dumps({"files": {"setup.bat": "sha256:" + __import__("hashlib").sha256(b"setup").hexdigest()}}),
        encoding="utf-8",
    )

    errors = verify_transcript(assets, assets / "checksums.json", assets / "release-verification-v1.txt", strict=True)

    assert any("self-hash" in error for error in errors)
    assert any("transcript hash mismatch for setup.bat" in error for error in errors)


def test_verify_release_transcript_uses_detached_manifest_for_transcript_hash(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "setup.bat").write_bytes(b"setup")
    (assets / "checksums.json").write_text(
        json.dumps({"files": {"setup.bat": "sha256:" + __import__("hashlib").sha256(b"setup").hexdigest()}}),
        encoding="utf-8",
    )
    transcript = assets / "release-verification-v1.txt"
    transcript.write_text(
        "Easy ASR Bench release verification transcript\n"
        "downloaded_assets:\n"
        f"  setup.bat sha256:{__import__('hashlib').sha256(b'setup').hexdigest()}\n",
        encoding="utf-8",
    )
    manifest_path = assets / "release-verification-manifest-v1.json"
    manifest_path.write_text(json.dumps(build_manifest("v1", transcript, [assets / "setup.bat"])), encoding="utf-8")

    assert verify_transcript(assets, assets / "checksums.json", transcript, strict=True, detached_manifest_path=manifest_path) == []

    transcript.write_text(transcript.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
    errors = verify_transcript(assets, assets / "checksums.json", transcript, strict=True, detached_manifest_path=manifest_path)

    assert any("detached verification manifest transcript hash mismatch" in error for error in errors)
