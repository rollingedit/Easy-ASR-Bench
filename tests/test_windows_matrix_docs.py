from pathlib import Path
import json


def test_windows_matrix_scripts_define_required_rows():
    script = Path("qa/windows_matrix/run_release_matrix.ps1").read_text(encoding="utf-8")
    required = json.loads(Path("tests/fixtures/release_required_rows_v2.json").read_text(encoding="utf-8"))["rows"]
    manual_matrix = json.loads(Path("qa/release_manual_rows_v2.json").read_text(encoding="utf-8"))["manual_matrix"]

    rows = set()
    for key, value in manual_matrix.items():
        if isinstance(value, dict):
            rows.update(value)
        else:
            rows.add(key)

    assert "qa\\release_manual_rows_v2.json" in script
    assert "Get-ManualRows" in script
    assert "win11_clean_no_python_setup" in rows
    assert "gguf_asr_mmproj_pair" in rows
    assert "dependency_install_declined" in rows
    assert "real_tiny_faster_whisper_report_smoke" in rows
    for row in required:
        assert row in rows
    assert 'Status "not_run"' in script


def test_windows_matrix_collector_writes_environment_and_hashes():
    script = Path("qa/windows_matrix/collect_release_evidence.ps1").read_text(encoding="utf-8")

    assert "environment_summary" in script
    assert "Get-FileHash" in script
    assert "row.json" in script
    assert "logs_sha256" in script
    assert "results_sha256" in script
    assert "Rows marked pass must include -AppVersion and -ReleaseCommit" in script


def test_public_asset_smoke_runner_captures_installed_app_json_evidence():
    script = Path("qa/windows_matrix/run_public_asset_smoke.ps1").read_text(encoding="utf-8")

    assert "gh release download $Tag --repo $Repo" in script
    assert "$WorkDir = (Resolve-Path -LiteralPath $WorkDir).Path" in script
    assert "$Output = (Resolve-Path -LiteralPath $Output).Path" in script
    assert "$AssetDir = (Resolve-Path -LiteralPath $AssetDir).Path" in script
    assert "setup.bat" in script
    assert "call `\"$setup`\" --dry-run --verify-release --asset-dir" in script
    assert "--dry-run --verify-release --asset-dir" in script
    assert "--doctor --json" in script
    assert "--first-run-smoke" in script
    assert "doctor.json" in script
    assert "first-run-smoke.json" in script
    assert "ConvertFrom-Json" in script
    assert "setup_dry_run_verify_release" in script
    assert "empty_models_guided_first_run" in script
    assert "Copy-Item -LiteralPath $verifyTranscript" in script
    assert "setup-verify-release.log" in script
    assert "Copy-Item -LiteralPath $doctorJson" in script
    assert "setup-install.log" in script
    assert "collect_release_evidence failed for setup_dry_run_verify_release" in script
    assert "collect_release_evidence failed for empty_models_guided_first_run" in script
    assert "-Status \"pass\"" in script
    assert "-ReleaseCommit $ReleaseCommit" in script


def test_release_verification_documents_real_tiny_model_smoke():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\run_real_tiny_model_smoke.py" in text
    assert "non-empty transcript" in text
    assert "VRAM measurement source" in text


def test_release_verification_documents_repair_all_safe_runtime_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row setup_repair_all_safe" in text
    assert "repair_plan.json" in text
    assert "repair_all_safe.json" in text
    assert "--install-deps" in text


def test_release_verification_documents_clean_vm_bootstrap_runtime_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row clean_vm_zero_dependency_bootstrap" in text
    assert "EASY_ASR_BENCH_CLEAN_VM_BOOTSTRAP_PROOF=1" in text
    assert "same-media multi-model SmolLM benchmark" in text


def test_release_verification_documents_validate_real_smoke_doctor_mode():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")
    setup_text = Path("docs/what_setup_installs.md").read_text(encoding="utf-8")

    assert "python -m app.doctor --config config.json --validate-real-smoke" in text
    assert "easy_asr_bench.real_smoke_validation.v1" in text
    assert "setup.bat --doctor --validate-real-smoke" in setup_text
    assert "setup_repair_all_safe" in setup_text
    assert "cpu_model_smoke" in setup_text
    assert "--no-network" in setup_text
    assert "--allow-downloads" in text
