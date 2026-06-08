import json
from pathlib import Path

import pytest

from app.merge_reference_parts import merge_parts
from app.reference_schema import validate_llm_reference
from app.utils import expand_inputs


def test_publish_release_workflow_builds_assets_on_github_from_draft():
    text = Path(".github/workflows/publish-release.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "python scripts/build_release_zip.py --version" in text
    assert "--update-metadata" in text
    assert "git diff --exit-code -- setup.bat installer/manifest.json installer/checksums.json" in text
    assert "python scripts/validate_physical_files.py --repo ." in text
    assert "python scripts/write_release_notes.py" in text
    assert "gh release create" in text
    assert "--draft" in text
    assert "gh release upload" in text
    assert "install.ps1" in text
    assert "Verify uploaded draft release assets" in text
    assert "scripts/verify_github_release.py" in text
    assert "Publish verified release" in text
    assert "prerelease" in text
    assert "allow_incomplete_smoke" in text
    assert "setup.bat --dry-run --verify-release" in text
    assert text.index("Verify uploaded draft release assets") < text.index("Publish verified release")
    assert text.index("gh release upload") < text.index("Verify uploaded draft release assets")


def test_bump_version_script_exists_and_checks_stale_versions():
    text = Path("scripts/bump_version.py").read_text(encoding="utf-8")

    assert "assert_old_version_removed" in text
    assert "setup.bat" in text
    assert "installer/install.ps1" in text


def test_release_version_coherence_script_exists():
    text = Path("scripts/check_release_version_coherence.py").read_text(encoding="utf-8")

    assert "app/version.py VERSION mismatch" in text
    assert "app.__version__ must come from app.version.VERSION" in text
    assert "setup.bat APP_VERSION mismatch" in text
    assert "installer default Version mismatch" in text
    assert "manifest app_zip mismatch" in text


def test_release_notes_script_writes_notes_file(tmp_path: Path):
    from scripts.write_release_notes import write_notes

    output = tmp_path / "notes.md"
    smoke = tmp_path / "smoke.json"
    smoke.write_text(
        '{"manual_rows":[{"id":"compare_html_offline","status":"pass"},{"id":"nvidia_cuda","status":"not_run"}]}',
        encoding="utf-8",
    )
    write_notes("v0.2.6", output, smoke)
    text = output.read_text(encoding="utf-8")

    assert "## What changed" in text
    assert "Release status: automated packaging checks may be present" in text
    assert "## Automated Packaging Checks" in text
    assert "## Manual Smoke Rows Marked Pass" in text
    assert "`compare_html_offline`: pass" in text
    assert "## Not Verified In Release Smoke" in text
    assert "`nvidia_cuda`: not_run" in text
    assert "## Known limits" in text
    assert "Strict all-pass" not in text
    assert "Unit tests passed." not in text


def test_release_notes_do_not_promote_not_claimable_changelog_bullets(tmp_path: Path):
    from scripts.write_release_notes import write_notes

    output = tmp_path / "notes.md"
    smoke = tmp_path / "smoke.json"
    smoke.write_text(
        '{"manual_rows":[{"id":"compare_html_offline","status":"not_run"}]}',
        encoding="utf-8",
    )

    write_notes("v0.3.5", output, smoke)
    text = output.read_text(encoding="utf-8")
    what_changed = text.split("## What changed", 1)[1].split("## Automated Packaging Checks", 1)[0]

    assert "Still not claimable" not in what_changed
    assert "Strict all-pass release smoke matrix" not in what_changed


def test_release_validator_parses_workflow_yaml():
    text = Path("scripts/validate_release_files.py").read_text(encoding="utf-8")

    assert "yaml.safe_load" in text
    assert "workflows" in text
    assert "validate_root(ROOT)" in text
    assert "validate_version_coherence" in text


def test_release_gate_requires_strict_committed_checksums():
    text = Path(".github/workflows/release-gate.yml").read_text(encoding="utf-8")

    assert "python scripts/build_release_zip.py --version $version --strict-checksums" in text
    assert "Verify public setup path" in text
    assert "setup.bat --dry-run --verify-release" in text


def test_reference_validation_rejects_source_hash_mismatch():
    results = {
        "source": {"sha256": "expected"},
        "chunk_plan": {"chunks": [{"chunk_id": "0001", "start_seconds": 0, "end_seconds": 1}]},
    }
    reference = {
        "schema": "easy_asr_bench.llm_reference.v1",
        "source_sha256": "wrong",
        "reference_type": "llm_corrected_reference",
        "segments": [{"chunk_id": "0001", "start_seconds": 0, "end_seconds": 1, "text": "hello", "uncertain": []}],
        "global_notes": [],
    }

    assert any("source_sha256" in error for error in validate_llm_reference(reference, results))


def test_merge_reference_parts_validates_source_consistency():
    part_a = {
        "schema": "easy_asr_bench.llm_reference.v1",
        "source_sha256": "a",
        "reference_type": "llm_corrected_reference",
        "segments": [],
        "global_notes": [],
    }
    part_b = dict(part_a, source_sha256="b")

    with pytest.raises(ValueError, match="source_sha256"):
        merge_parts([part_a, part_b])


def test_expand_inputs_can_report_unsupported_files(tmp_path: Path):
    good = tmp_path / "good.wav"
    bad = tmp_path / "bad.xyz"
    good.write_text("", encoding="utf-8")
    bad.write_text("", encoding="utf-8")

    files, skipped = expand_inputs([tmp_path], {".wav"}, recursive=True, include_skipped=True)

    assert files == [good]
    assert skipped == [bad]


def test_installer_secondary_safety_fixes_present():
    text = Path("installer/install.ps1").read_text(encoding="utf-8")

    assert "function Invoke-Download" in text
    assert "(Test-Path (Join-Path $InstallDir \"app\\doctor.py\")) -and (Test-Path $python)" in text
    assert "Remove-Item -LiteralPath $Backup" in text


def test_vram_metrics_are_sampled_during_model_runs():
    text = Path("app/main.py").read_text(encoding="utf-8")

    assert "reset_peak_vram()" in text
    assert "peak_vram_sample" in text
    assert "result.metrics.update(peak_vram_sample())" in text
