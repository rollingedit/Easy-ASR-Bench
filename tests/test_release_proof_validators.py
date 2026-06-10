import json
from pathlib import Path

from scripts.validate_release_smoke import validate_smoke
from scripts.validate_v040_release_readiness import validate_readiness
from scripts.merge_release_evidence import evidence_rows, merge_manual_rows
from scripts.collect_win10_setup_evidence import validate_win10_row
from scripts.verify_release_transcript import verify_transcript
from scripts.write_release_verification_manifest import build_manifest


def test_validate_release_smoke_fails_missing_required_row():
    errors = validate_smoke({"schema": "easy_asr_bench.release_smoke.v2", "manual_rows": []}, ["win11_clean_no_python_setup"])

    assert any("missing required smoke row" in error for error in errors)


def test_validate_release_smoke_requires_pass_when_strict():
    smoke = {"schema": "easy_asr_bench.release_smoke.v2", "manual_rows": [{"id": "win11_clean_no_python_setup", "status": "not_run"}]}

    errors = validate_smoke(smoke, ["win11_clean_no_python_setup"], require_all_pass=True)

    assert "win11_clean_no_python_setup status is not_run, expected pass" in errors


def test_validate_release_smoke_allows_blocked_with_external_requirement_when_not_strict():
    smoke = {
        "schema": "easy_asr_bench.release_smoke.v2",
        "manual_rows": [
            {
                "id": "nvidia_cuda_torch_onnx_faster_whisper_llama",
                "status": "blocked",
                "block_reason": "No NVIDIA GPU was detected.",
                "external_requirement": "NVIDIA CUDA GPU",
            }
        ],
    }

    assert validate_smoke(smoke, ["nvidia_cuda_torch_onnx_faster_whisper_llama"]) == []


def test_validate_release_smoke_requires_blocked_external_requirement():
    smoke = {
        "schema": "easy_asr_bench.release_smoke.v2",
        "manual_rows": [{"id": "nvidia_cuda_torch_onnx_faster_whisper_llama", "status": "blocked"}],
    }

    errors = validate_smoke(smoke, ["nvidia_cuda_torch_onnx_faster_whisper_llama"])

    assert "nvidia_cuda_torch_onnx_faster_whisper_llama blocked row is missing block_reason" in errors
    assert "nvidia_cuda_torch_onnx_faster_whisper_llama blocked row is missing external_requirement" in errors


def test_validate_release_smoke_requires_version_and_commit_when_strict():
    smoke = {"schema": "easy_asr_bench.release_smoke.v2", "manual_rows": [{"id": "win11_clean_no_python_setup", "status": "pass"}]}

    errors = validate_smoke(smoke, ["win11_clean_no_python_setup"], require_all_pass=True)

    assert "win11_clean_no_python_setup is missing app_version" in errors
    assert "win11_clean_no_python_setup is missing release_commit" in errors
    assert "win11_clean_no_python_setup is missing execution_git_commit" in errors


def test_validate_release_smoke_rejects_stale_execution_commit_when_strict():
    smoke = {
        "schema": "easy_asr_bench.release_smoke.v2",
        "commit": "release123",
        "manual_rows": [
            {
                "id": "win11_clean_no_python_setup",
                "status": "pass",
                "app_version": "v0.4.0",
                "release_commit": "old123",
                "target_release_commit": "release123",
                "execution_git_commit": "old123",
                "execution_git_dirty": False,
            }
        ],
    }

    errors = validate_smoke(smoke, ["win11_clean_no_python_setup"], require_all_pass=True)

    assert "win11_clean_no_python_setup execution_git_commit old123 does not match release commit release123" in errors


def test_validate_release_smoke_allows_documented_external_execution_commit_exception_when_strict():
    smoke = {
        "schema": "easy_asr_bench.release_smoke.v2",
        "commit": "release123",
        "manual_rows": [
            {
                "id": "nvidia_cuda_torch_onnx_faster_whisper_llama",
                "status": "pass",
                "app_version": "v0.4.0",
                "release_commit": "external123",
                "target_release_commit": "release123",
                "execution_git_commit": "external123",
                "execution_git_dirty": False,
                "execution_commit_exception": "External hardware row ran from an immutable release-candidate artifact before final history rewrite.",
                "results_sha256": "sha256:result",
                "environment_summary": {"system": "Windows"},
            }
        ],
    }

    errors = validate_smoke(
        smoke,
        ["nvidia_cuda_torch_onnx_faster_whisper_llama"],
        require_all_pass=True,
        require_log_hashes=True,
        require_environment_summary=True,
    )

    assert errors == []


def test_validate_release_smoke_requires_evidence_fields_when_strict():
    smoke = {"schema": "easy_asr_bench.release_smoke.v2", "manual_rows": [{"id": "win11_clean_no_python_setup", "status": "pass"}]}

    errors = validate_smoke(smoke, ["win11_clean_no_python_setup"], require_log_hashes=True, require_environment_summary=True)

    assert any("logs_sha256" in error for error in errors)
    assert any("environment_summary" in error for error in errors)


def test_collect_win10_setup_evidence_accepts_real_win10_shape():
    row = {
        "id": "win10_existing_python_setup",
        "status": "pass",
        "details": {
            "platform": {"system": "Windows", "release": "10", "version": "10.0.19045"},
            "python_probe": {"python_visible_on_path": True},
            "setup_static_contract": {"missing_markers": []},
            "setup_dry_run_local": {"exit_code": 0},
        },
    }

    assert validate_win10_row(row) == []


def test_collect_win10_setup_evidence_rejects_windows11_build():
    row = {
        "id": "win10_existing_python_setup",
        "status": "pass",
        "details": {
            "platform": {"system": "Windows", "release": "10", "version": "10.0.22631"},
            "python_probe": {"python_visible_on_path": True},
            "setup_static_contract": {"missing_markers": []},
            "setup_dry_run_local": {"exit_code": 0},
        },
    }

    errors = validate_win10_row(row)

    assert any("Windows 11 build floor" in error for error in errors)


def test_v040_readiness_reports_strict_smoke_and_history_blockers(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    import subprocess

    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    (repo / "note.txt").write_text("NVIDIA GeForce RTX " + "40" + "90" + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "note.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "dirty history"], cwd=repo, check=True, capture_output=True)
    (repo / "note.txt").write_text("generic validation host\n", encoding="utf-8")
    subprocess.run(["git", "add", "note.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "clean tip"], cwd=repo, check=True, capture_output=True)
    smoke = tmp_path / "smoke.json"
    smoke.write_text(
        json.dumps(
            {
                "schema": "easy_asr_bench.release_smoke.v2",
                "manual_rows": [
                    {
                        "id": "win10_existing_python_setup",
                        "status": "blocked",
                        "block_reason": "needs Windows 10",
                        "external_requirement": "Windows 10 VM",
                        "environment_summary": {"system": "Windows"},
                        "results_sha256": "sha256:test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    required = tmp_path / "required.json"
    required.write_text(json.dumps({"rows": ["win10_existing_python_setup"]}), encoding="utf-8")

    errors = validate_readiness(smoke=smoke, required=required, repo=repo, publish_ref="main")

    assert any(error.startswith("strict smoke: win10_existing_python_setup status is blocked") for error in errors)
    assert any(error.startswith("public history hygiene:") for error in errors)


def test_merge_release_evidence_replaces_matching_manual_rows(tmp_path: Path):
    row_dir = tmp_path / "evidence" / "setup_dry_run_verify_release"
    row_dir.mkdir(parents=True)
    row = {
        "id": "setup_dry_run_verify_release",
        "status": "pass",
        "app_version": "v0.3.7",
        "release_commit": "abc123",
        "logs_sha256": "sha256:log",
        "results_sha256": "",
        "environment_summary": {"os": "Windows"},
    }
    (row_dir / "row.json").write_text(json.dumps(row), encoding="utf-8")

    merged = merge_manual_rows(
        {
            "schema": "easy_asr_bench.release_smoke.v2",
            "manual_rows": [
                {"id": "setup_dry_run_verify_release", "status": "not_run"},
                {"id": "empty_models_guided_first_run", "status": "not_run"},
            ],
        },
        evidence_rows(tmp_path / "evidence"),
    )

    assert merged["manual_rows"][0]["status"] == "pass"
    assert merged["manual_rows"][0]["app_version"] == "v0.3.7"
    assert merged["manual_rows"][1]["status"] == "not_run"
    assert merged["manual_row_status_counts"] == {"pass": 1, "not_run": 1}


def test_merge_release_evidence_updates_matrix_and_preserves_blocked_rows(tmp_path: Path):
    row_dir = tmp_path / "evidence" / "real_media_download_cache"
    row_dir.mkdir(parents=True)
    blocked = {
        "id": "real_media_download_cache",
        "status": "blocked",
        "block_reason": "network disabled",
        "external_requirement": "rerun with --include-network --allow-downloads",
    }
    (row_dir / "row.json").write_text(json.dumps(blocked), encoding="utf-8")

    merged = merge_manual_rows(
        {
            "schema": "easy_asr_bench.release_smoke.v2",
            "manual_matrix": {
                "media": {
                    "real_media_download_cache": "not_run",
                    "wav_mp3_mp4_media": "not_run",
                }
            },
            "manual_rows": [
                {"id": "real_media_download_cache", "status": "not_run"},
                {"id": "wav_mp3_mp4_media", "status": "not_run"},
            ],
        },
        evidence_rows(tmp_path / "evidence"),
    )

    assert merged["manual_rows"][0]["status"] == "blocked"
    assert merged["manual_matrix"]["media"]["real_media_download_cache"] == "blocked"
    assert merged["manual_matrix"]["media"]["wav_mp3_mp4_media"] == "not_run"
    assert merged["manual_row_status_counts"] == {"blocked": 1, "not_run": 1}
    assert merged["blocked_rows"] == [
        {
            "id": "real_media_download_cache",
            "block_reason": "network disabled",
            "external_requirement": "rerun with --include-network --allow-downloads",
        }
    ]


def test_merge_release_evidence_prefers_external_pass_but_keeps_local_blocked_variant(tmp_path: Path):
    local_dir = tmp_path / "evidence" / "local" / "torch_cuda_tensor_smoke"
    external_dir = tmp_path / "evidence" / "external_cuda" / "torch_cuda_tensor_smoke"
    local_dir.mkdir(parents=True)
    external_dir.mkdir(parents=True)
    local_blocked = {
        "id": "torch_cuda_tensor_smoke",
        "status": "blocked",
        "block_reason": "missing NVIDIA CUDA-capable GPU",
        "external_requirement": "NVIDIA CUDA machine with a CUDA-enabled Torch wheel",
        "environment": {"machine": "local-non-nvidia"},
    }
    external_pass = {
        "id": "torch_cuda_tensor_smoke",
        "status": "pass",
        "environment": {"machine": "external-cuda"},
        "details": {"tensor_device": "cuda:0"},
    }
    (local_dir / "row.json").write_text(json.dumps(local_blocked), encoding="utf-8")
    (external_dir / "row.json").write_text(json.dumps(external_pass), encoding="utf-8")

    merged = merge_manual_rows(
        {
            "schema": "easy_asr_bench.release_smoke.v2",
            "manual_matrix": {"provider_smoke": {"torch_cuda_tensor_smoke": "not_run"}},
            "manual_rows": [{"id": "torch_cuda_tensor_smoke", "status": "not_run"}],
        },
        evidence_rows(tmp_path / "evidence"),
    )

    row = merged["manual_rows"][0]
    assert row["status"] == "pass"
    assert row["details"]["tensor_device"] == "cuda:0"
    assert merged["manual_matrix"]["provider_smoke"]["torch_cuda_tensor_smoke"] == "pass"
    assert {variant["status"] for variant in row["merged_evidence_variants"]} == {"blocked", "pass"}
    assert any(variant.get("block_reason") == "missing NVIDIA CUDA-capable GPU" for variant in row["merged_evidence_variants"])


def test_merge_release_evidence_prefers_newest_duplicate_at_same_status(tmp_path: Path):
    old_dir = tmp_path / "evidence" / "old" / "win10_existing_python_setup"
    new_dir = tmp_path / "evidence" / "new" / "win10_existing_python_setup"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    old_row = {
        "id": "win10_existing_python_setup",
        "status": "blocked",
        "created_utc": "2026-06-09T00:00:00+00:00",
        "block_reason": "old reason",
        "external_requirement": "Windows 10 VM",
    }
    new_row = {
        "id": "win10_existing_python_setup",
        "status": "blocked",
        "created_utc": "2026-06-10T00:00:00+00:00",
        "block_reason": "new reason",
        "external_requirement": "Windows 10 VM",
    }
    (old_dir / "row.json").write_text(json.dumps(old_row), encoding="utf-8")
    (new_dir / "row.json").write_text(json.dumps(new_row), encoding="utf-8")

    merged = merge_manual_rows(
        {
            "schema": "easy_asr_bench.release_smoke.v2",
            "manual_rows": [{"id": "win10_existing_python_setup", "status": "not_run"}],
        },
        evidence_rows(tmp_path / "evidence"),
    )

    row = merged["manual_rows"][0]
    assert row["block_reason"] == "new reason"
    assert {variant["block_reason"] for variant in row["merged_evidence_variants"]} == {"old reason", "new reason"}


def test_merge_release_evidence_fail_variant_dominates_duplicate_pass(tmp_path: Path):
    pass_dir = tmp_path / "evidence" / "pass" / "same_row"
    fail_dir = tmp_path / "evidence" / "fail" / "same_row"
    pass_dir.mkdir(parents=True)
    fail_dir.mkdir(parents=True)
    pass_row = {
        "id": "same_row",
        "status": "pass",
        "created_utc": "2026-06-10T00:00:00+00:00",
        "summary": "later pass",
    }
    fail_row = {
        "id": "same_row",
        "status": "fail",
        "created_utc": "2026-06-09T00:00:00+00:00",
        "summary": "older fail",
    }
    (pass_dir / "row.json").write_text(json.dumps(pass_row), encoding="utf-8")
    (fail_dir / "row.json").write_text(json.dumps(fail_row), encoding="utf-8")

    merged = merge_manual_rows(
        {"schema": "easy_asr_bench.release_smoke.v2", "manual_rows": [{"id": "same_row", "status": "not_run"}]},
        evidence_rows(tmp_path / "evidence"),
    )

    row = merged["manual_rows"][0]
    assert row["status"] == "fail"
    assert row["summary"] == "older fail"
    assert {variant["status"] for variant in row["merged_evidence_variants"]} == {"pass", "fail"}


def test_merge_release_evidence_reads_powershell_utf8_bom(tmp_path: Path):
    evidence = tmp_path / "evidence" / "row1"
    evidence.mkdir(parents=True)
    (evidence / "row.json").write_text(
        "\ufeff" + json.dumps({"id": "row1", "status": "pass"}),
        encoding="utf-8",
    )

    assert evidence_rows(tmp_path / "evidence")["row1"]["status"] == "pass"


def test_merge_release_evidence_rejects_unknown_rows():
    try:
        merge_manual_rows({"manual_rows": [{"id": "known", "status": "not_run"}]}, {"unknown": {"id": "unknown", "status": "pass"}})
    except SystemExit as exc:
        assert "not present in release smoke" in str(exc)
    else:
        raise AssertionError("unknown evidence row should fail")


def test_merge_release_evidence_can_ignore_unknown_rows():
    merged = merge_manual_rows(
        {"manual_rows": [{"id": "known", "status": "not_run"}]},
        {
            "known": {"id": "known", "status": "pass"},
            "unknown": {"id": "unknown", "status": "pass"},
        },
        ignore_unknown=True,
    )

    assert merged["manual_rows"] == [{"id": "known", "status": "pass"}]
    assert merged["manual_row_status_counts"] == {"pass": 1}


def test_evidence_rows_can_ignore_malformed_rows(tmp_path: Path):
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    good.mkdir()
    bad.mkdir()
    (good / "row.json").write_text(json.dumps({"id": "known", "status": "pass"}), encoding="utf-8")
    (bad / "row.json").write_text(json.dumps({"status": "blocked", "summary": "offline"}), encoding="utf-8")

    rows = evidence_rows(tmp_path, ignore_malformed=True)

    assert list(rows) == ["known"]


def test_merge_release_evidence_adds_strict_evidence_fields(tmp_path: Path):
    result = tmp_path / "results.json"
    result.write_text(json.dumps({"ok": True}), encoding="utf-8")
    merged = merge_manual_rows(
        {
            "tag": "v0.3.9",
            "commit": "abc123",
            "manual_rows": [{"id": "known", "status": "not_run"}],
        },
        {
            "known": {
                "id": "known",
                "status": "pass",
                "execution_git_commit": "abc123",
                "execution_git_dirty": False,
                "environment": {"system": "Windows", "python": "3.12"},
                "artifacts": [{"path": str(result), "sha256": "sha256:result"}],
            }
        },
    )

    row = merged["manual_rows"][0]
    assert row["app_version"] == "v0.3.9"
    assert row["release_commit"] == "abc123"
    assert row["target_release_commit"] == "abc123"
    assert row["execution_git_commit"] == "abc123"
    assert row["environment_summary"] == {"system": "Windows", "python": "3.12"}
    assert row["results_sha256"] == "sha256:result"


def test_runtime_matrix_write_row_records_execution_git_state(tmp_path: Path, monkeypatch):
    import qa.runtime_matrix.common as common

    monkeypatch.setattr(
        common,
        "git_state",
        lambda: {
            "execution_git_commit": "abc123",
            "execution_git_dirty": False,
            "execution_git_status": "clean",
        },
    )

    row = common.write_row("known", "pass", tmp_path, summary="ok")

    assert row["execution_git_commit"] == "abc123"
    assert row["execution_git_dirty"] is False
    assert row["execution_git_status"] == "clean"
    assert json.loads((tmp_path / "row.json").read_text(encoding="utf-8"))["execution_git_commit"] == "abc123"


def test_merge_release_evidence_sanitizes_public_machine_details(tmp_path: Path):
    row_dir = tmp_path / "row"
    row_dir.mkdir()
    user_cache = "C:" + "\\Users\\" + "PC" + "\\.cache\\huggingface\\hub"
    exact_gpu = "NVIDIA GeForce RTX " + "40" + "90"
    exact_cpu = "13th Gen Intel(R) Core(TM) " + "i5" + "-13600K"
    exact_igpu = "Intel(R) UHD Graphics " + "770"
    other_project = "E:" + "\\_github\\" + "diffusion-audio-" + "restoration-windows"
    row = {
        "id": "known",
        "status": "pass",
        "environment": {"cache_dir": user_cache},
        "details": {
            "cuda_provider_checks": {
                "torch_gpu_names": [exact_gpu],
                "messages": [f"cache path {user_cache}"],
            },
            "cpu": exact_cpu,
            "igpu": exact_igpu,
            "sandbox_command": f"{other_project}\\evidence\\sandbox.wsb <AccountPassword>secret</AccountPassword>",
        },
    }
    (row_dir / "row.json").write_text(json.dumps(row), encoding="utf-8")

    merged = merge_manual_rows(
        {
            "schema": "easy_asr_bench.release_smoke.v2",
            "tag": "v0.4.0",
            "commit": "abc123",
            "manual_rows": [{"id": "known", "status": "not_run"}],
        },
        evidence_rows(tmp_path),
    )

    text = json.dumps(merged)
    assert exact_gpu not in text
    assert user_cache.replace("\\", "\\\\") not in text
    assert exact_cpu not in text
    assert exact_igpu not in text
    assert "diffusion-audio-" + "restoration" not in text
    assert "secret" not in text
    assert "NVIDIA CUDA GPU" in text
    assert "%USERPROFILE%" in text
    assert "%LOCAL_WORKSPACE%" in text
    assert "<redacted>" in text


def test_merge_release_evidence_hashes_row_file_when_no_result_artifact(tmp_path: Path):
    evidence = tmp_path / "row.json"
    evidence.write_text(json.dumps({"id": "known", "status": "pass"}), encoding="utf-8")
    rows = evidence_rows(tmp_path)
    merged = merge_manual_rows({"manual_rows": [{"id": "known", "status": "not_run"}]}, rows)

    assert merged["manual_rows"][0]["results_sha256"].startswith("sha256:")


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
