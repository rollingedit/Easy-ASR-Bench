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


def test_release_verification_documents_setup_dry_run_json_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "cmd /c setup.bat --dry-run --local --json" in text
    assert "qa\\runtime_matrix\\run_row.py --row setup_dry_run_json" in text
    assert "easy_asr_bench.setup_dry_run.v1" in text
    assert "no_files_modified=true" in text


def test_release_verification_documents_vc_runtime_repair_contract_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row windows_vc_runtime_repair_contract" in text
    assert "Microsoft.VCRedist.2015+.x64" in text
    assert "--accept-package-agreements" in text
    assert "post-repair redistributable probe passes" in text


def test_release_verification_documents_python_packaging_repair_contract_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row python_packaging_tools_repair_contract" in text
    assert "requirements\\python_packaging.txt" in text
    assert "app.repair_plan.execute_repair_plan" in text
    assert "pkg_resources" in text


def test_release_verification_documents_transformers_dependency_repair_contract_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row transformers_cpu_dependency_repair_contract" in text
    assert "requirements\\transformers_cpu.txt" in text
    assert "torch`, `transformers`, `safetensors`, `sentencepiece`, `google.protobuf`, and `torchaudio`" in text
    assert "runtime-resolution record" in text


def test_release_verification_documents_media_tools_repair_contract_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row media_tools_dependency_repair_contract" in text
    assert "requirements\\core.txt" in text
    assert "imageio_ffmpeg" in text
    assert "media-tools FFmpeg runtime" in text
    assert "real audio/video smoke rows" in text


def test_release_verification_documents_llama_mtmd_repair_contract_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row llama_mtmd_dependency_repair_contract" in text
    assert "llama-mtmd-cli" in text
    assert "llama_mtmd_runtime_probe" in text
    assert "native_tool" in text
    assert "ASR GGUF+mmproj rows" in text


def test_release_verification_documents_directml_conflict_repair_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row directml_provider_conflict_repair" in text
    assert "install_group_for_config(\"onnx\"" in text
    assert "onnxruntime-directml" in text
    assert "commands captured instead of executed" in text


def test_release_verification_documents_transformers_cuda_fallback_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row transformers_cuda_unavailable_cpu_fallback" in text
    assert "requirements\\torch_cuda_cu128.txt" in text
    assert "requirements\\transformers_cpu.txt" in text
    assert "conservative safe recovery command" in text
    assert "explicit CPU fallback" in text


def test_release_verification_documents_openai_whisper_cuda_fallback_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row openai_whisper_cuda_unavailable_cpu_fallback" in text
    assert "requirements\\torch_cuda_cu128.txt" in text
    assert "requirements\\openai_whisper.txt" in text
    assert "OpenAI Whisper `.pt` CUDA fallback" in text


def test_release_verification_documents_generic_onnx_openvino_fallback_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row generic_onnx_openvino_unavailable_cpu_fallback" in text
    assert "requested_runtime_provider=openvino" in text
    assert "openvino_requested=true" in text
    assert "provider_fallback=true" in text


def test_release_verification_documents_faster_whisper_pkg_resources_repair_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row faster_whisper_pkg_resources_repair" in text
    assert "pkg_resources" in text
    assert "requirements\\faster_whisper.txt" in text
    assert "native load probe is rerun" in text


def test_release_verification_documents_faster_whisper_candidate_fallback_repair_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row faster_whisper_ctranslate2_candidate_fallback_repair" in text
    assert "discovers CTranslate2 versions dynamically" in text
    assert "skips versions outside `requirements\\faster_whisper.txt`" in text
    assert "not a hard-coded primary install" in text


def test_release_verification_documents_faster_whisper_vc_runtime_repair_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row faster_whisper_vc_runtime_repair" in text
    assert "CTranslate2 missing-DLL native-load failures" in text
    assert "before replacing CTranslate2 packages" in text


def test_release_verification_documents_model_layout_repair_runtime_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row setup_repair_model_layouts" in text
    assert "hf_model_layout_repair_plan.json" in text
    assert "model_layout_repair_sweep.json" in text
    assert "last_execution" in text


def test_release_verification_documents_clean_vm_bootstrap_runtime_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row clean_vm_zero_dependency_bootstrap" in text
    assert "EASY_ASR_BENCH_CLEAN_VM_BOOTSTRAP_PROOF=1" in text
    assert "setup_repair_model_layouts` subrow" in text
    assert "same-media multi-model SmolLM benchmark" in text


def test_release_verification_documents_repair_plan_issue_classification_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row repair_plan_issue_classification_contract" in text
    assert "corrupt installs" in text
    assert "incompatible package/runtime stacks" in text


def test_release_verification_documents_stale_cached_resolution_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row repair_all_safe_stale_cached_resolution" in text
    assert "cached-resolution repair contract" in text
    assert "does not reinstall packages unnecessarily" in text


def test_release_verification_documents_report_atomic_write_cleanup_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row report_atomic_write_failure_cleanup" in text
    assert "failed `.partial` files are removed" in text
    assert "previous complete artifact remains intact" in text


def test_release_verification_documents_watched_folder_queue_contract_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row watched_folder_partial_write_queue_contract" in text
    assert "waits through a partial write" in text
    assert "completed fast keys are skipped without rehashing" in text


def test_release_verification_documents_model_fixture_quality_claims_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row model_fixture_quality_claims" in text
    assert "structural tiny/random/generated fixtures" in text
    assert "not quality-bearing" in text


def test_release_verification_documents_hf_downloader_package_variant_row():
    text = Path("docs/release_verification.md").read_text(encoding="utf-8")

    assert "qa\\runtime_matrix\\run_row.py --row hf_downloader_package_variant_taxonomy" in text
    assert "sharded Safetensors indexes" in text
    assert "ONNX precision/quant variants" in text
    assert "split GGUF parts stay grouped" in text


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


def test_setup_docs_include_python_packaging_repair_group():
    text = Path("docs/what_setup_installs.md").read_text(encoding="utf-8")

    assert "Python packaging repair tools" in text
    assert "requirements/python_packaging.txt" in text
    assert "pip" in text
    assert "setuptools" in text
    assert "pkg_resources" in text
    assert "dependency_resolution_<group>.json" in text
