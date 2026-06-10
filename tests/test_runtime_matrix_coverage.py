import json
import os
from pathlib import Path

import pytest

from qa.runtime_matrix.registry import ROWS


ROOT = Path(__file__).resolve().parents[1]


def _manual_row_ids() -> set[str]:
    data = json.loads((ROOT / "qa" / "release_manual_rows_v2.json").read_text(encoding="utf-8"))
    ids: set[str] = set()
    for key, value in data["manual_matrix"].items():
        if isinstance(value, dict):
            ids.update(value)
        else:
            ids.add(key)
    return ids


def test_every_release_manual_row_has_runtime_matrix_script_definition():
    assert _manual_row_ids() <= set(ROWS)


def test_release_manual_matrix_includes_repair_all_safe_row():
    assert "setup_repair_all_safe" in _manual_row_ids()
    assert "repair_all_safe_failure_isolation" in _manual_row_ids()
    assert "repair_all_safe_stale_cached_resolution" in _manual_row_ids()
    assert "repair_plan_issue_classification_contract" in _manual_row_ids()


def test_every_required_release_row_has_runtime_matrix_script_definition():
    data = json.loads((ROOT / "tests" / "fixtures" / "release_required_rows_v2.json").read_text(encoding="utf-8"))
    assert set(data["rows"]) <= set(ROWS)


def test_required_release_rows_include_bootstrapper_diagnostics_and_repair():
    data = json.loads((ROOT / "tests" / "fixtures" / "release_required_rows_v2.json").read_text(encoding="utf-8"))

    assert "setup_doctor_strict" in data["rows"]
    assert "setup_repair_all_safe" in data["rows"]
    assert "repair_all_safe_failure_isolation" in data["rows"]
    assert "repair_all_safe_stale_cached_resolution" in data["rows"]
    assert "repair_plan_issue_classification_contract" in data["rows"]
    assert "setup_repair_model_layouts" in data["rows"]


def test_required_release_rows_include_same_media_smollm_benchmarks():
    data = json.loads((ROOT / "tests" / "fixtures" / "release_required_rows_v2.json").read_text(encoding="utf-8"))

    assert "same_media_multi_model_smollm_benchmark" in data["rows"]
    assert "same_media_multi_model_smollm_benchmark_directml" in data["rows"]


def test_runtime_matrix_local_runner_plans_local_rows_without_network(tmp_path):
    from qa.runtime_matrix.run_all_local import write_plan

    plan = write_plan(tmp_path, include_network=False, include_external=False)
    actions = {row["id"]: row["action"] for row in plan["rows"]}

    assert actions["hf_downloader_qwen3_asr_gguf_mmproj_public_real_download_to_asr"] == "blocked_without_network"
    assert actions["win11_clean_no_python_setup"] == "run_for_blocked_evidence"
    assert actions["clean_vm_zero_dependency_bootstrap"] == "run_for_blocked_evidence"
    assert actions["nvidia_cuda_torch_onnx_faster_whisper_llama"] == "run_for_blocked_evidence"
    assert actions["setup_doctor_strict"] == "run"
    assert (tmp_path / "plan.json").exists()


def test_runtime_matrix_local_runner_can_plan_selected_rows(tmp_path):
    from qa.runtime_matrix.run_all_local import write_plan

    plan = write_plan(tmp_path, include_network=False, include_external=False, selected_rows={"setup_doctor_strict"})

    assert [row["id"] for row in plan["rows"]] == ["setup_doctor_strict"]
    assert plan["selected_rows"] == ["setup_doctor_strict"]


def test_runtime_matrix_local_runner_writes_blocked_network_row(tmp_path):
    from qa.runtime_matrix.run_all_local import _write_network_blocked

    result = _write_network_blocked("real_media_download_cache", tmp_path)
    row = json.loads((tmp_path / "real_media_download_cache" / "row.json").read_text(encoding="utf-8"))

    assert result["status"] == "blocked"
    assert row["status"] == "blocked"
    assert row["external_requirement"] == "rerun with --include-network --allow-downloads"


def test_runtime_matrix_local_powershell_wrapper_exists():
    script = ROOT / "qa" / "runtime_matrix" / "run_all_local.ps1"
    text = script.read_text(encoding="utf-8")

    assert "run_all_local.py" in text
    assert "--allow-downloads" in text
    assert "--include-network" in text
    assert "--row" in text


def test_runtime_matrix_rows_point_to_importable_modules():
    for definition in ROWS.values():
        __import__(definition.module, fromlist=["run"])


def test_runtime_matrix_includes_native_runtime_prerequisite_rows():
    assert ROWS["windows_vc_runtime"].module == "qa.runtime_matrix.rows.windows_vc_runtime"
    assert ROWS["windows_vc_runtime_repair_contract"].module == "qa.runtime_matrix.rows.windows_vc_runtime"
    assert ROWS["python_packaging_tools_repair_contract"].module == "qa.runtime_matrix.rows.python_packaging_repair"
    assert ROWS["transformers_cpu_dependency_repair_contract"].module == "qa.runtime_matrix.rows.transformers_dependency_repair"
    assert ROWS["media_tools_dependency_repair_contract"].module == "qa.runtime_matrix.rows.media_tools_repair"
    assert ROWS["llama_mtmd_dependency_repair_contract"].module == "qa.runtime_matrix.rows.llama_mtmd_repair"
    assert ROWS["windows_directml_provider"].module == "qa.runtime_matrix.rows.windows_directml_provider"
    assert ROWS["directml_provider_conflict_repair"].module == "qa.runtime_matrix.rows.windows_directml_provider"
    assert ROWS["windows_vulkan_runtime"].module == "qa.runtime_matrix.rows.windows_vulkan_runtime"
    assert ROWS["amd_directml_onnx_smoke"].module == "qa.runtime_matrix.rows.generic_onnx_ctc_tiny"
    assert ROWS["intel_openvino_onnx_smoke"].module == "qa.runtime_matrix.rows.generic_onnx_ctc_tiny"
    assert ROWS["nvidia_cuda_torch_onnx_faster_whisper_llama"].module == "qa.runtime_matrix.rows.cuda_provider_matrix"
    assert ROWS["nvidia_cuda_hardware_detection"].module == "qa.runtime_matrix.rows.cuda_provider_matrix"
    assert ROWS["torch_cuda_tensor_smoke"].module == "qa.runtime_matrix.rows.cuda_provider_matrix"
    assert ROWS["onnxruntime_cuda_tiny_session"].module == "qa.runtime_matrix.rows.cuda_provider_matrix"
    assert ROWS["faster_whisper_ctranslate2_cuda_smoke"].module == "qa.runtime_matrix.rows.cuda_provider_matrix"
    assert ROWS["llama_cpp_cuda_smollm_smoke"].module == "qa.runtime_matrix.rows.cuda_provider_matrix"
    assert ROWS["llama_cpp_vulkan_smollm_smoke"].module == "qa.runtime_matrix.rows.windows_vulkan_runtime"
    assert ROWS["faster_whisper_cuda_unavailable_cpu_fallback"].module == "qa.runtime_matrix.rows.cuda_provider_matrix"
    assert ROWS["transformers_cuda_unavailable_cpu_fallback"].module == "qa.runtime_matrix.rows.cuda_provider_matrix"
    assert ROWS["vulkan_runtime_no_sdk"].module == "qa.runtime_matrix.rows.windows_vulkan_runtime"


def test_cuda_provider_matrix_parses_last_json_payload():
    from qa.runtime_matrix.rows.cuda_provider_matrix import _last_json_object

    text = 'progress {"nested": {"value": 1}}\n{"schema": "payload", "metrics": {"device": "cuda"}}\n'

    assert _last_json_object(text) == {"schema": "payload", "metrics": {"device": "cuda"}}


def test_clean_vm_bootstrap_parses_last_json_payload_from_mixed_stdout():
    from qa.runtime_matrix.rows.clean_vm_bootstrap import _json_from_command_stdout

    result = {
        "stdout_tail": (
            "Microsoft Visual C++ Redistributable is not installed\n"
            '{"schema": "intermediate"}\n'
            '{"schema": "easy_asr_bench.first_run_smoke.v1", "repair_plan_schema": "easy_asr_bench.repair_plan.v1"}\n'
        )
    }

    assert _json_from_command_stdout(result) == {
        "schema": "easy_asr_bench.first_run_smoke.v1",
        "repair_plan_schema": "easy_asr_bench.repair_plan.v1",
    }


def test_windows_vc_runtime_repair_contract_row_records_winget_command(tmp_path):
    from qa.runtime_matrix.rows import windows_vc_runtime

    row = windows_vc_runtime.run("windows_vc_runtime_repair_contract", tmp_path, False, False)

    assert row["status"] == "pass"
    command = row["details"]["commands"][0]
    assert command == row["details"]["expected_command"]
    assert command[:5] == ["winget", "install", "-e", "--id", "Microsoft.VCRedist.2015+.x64"]
    assert "--accept-package-agreements" in command
    assert "--accept-source-agreements" in command
    assert row["details"]["visual_cpp_redistributable"]["installed"] is True


def test_python_packaging_tools_repair_contract_row_records_shared_repair_contract(tmp_path):
    from qa.runtime_matrix.rows import python_packaging_repair

    row = python_packaging_repair.run("python_packaging_tools_repair_contract", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["missing_before"] == ["pkg_resources"]
    assert row["details"]["missing_after"] == []
    assert "requirements/python_packaging.txt" in row["details"]["repair_command"]
    assert row["details"]["backend_probe"]["kind"] == "python_import_probe"
    assert row["details"]["runtime_resolution_path"]


def test_media_tools_dependency_repair_contract_row_records_ffmpeg_runtime_resolution(tmp_path):
    from qa.runtime_matrix.rows import media_tools_repair

    row = media_tools_repair.run("media_tools_dependency_repair_contract", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["missing_before"] == ["imageio_ffmpeg", "ffmpeg executable"]
    assert row["details"]["missing_after"] == []
    assert "requirements/core.txt" in row["details"]["repair_command"]
    assert row["details"]["backend_probe"]["kind"] == "media_tools_ffmpeg_probe"
    assert row["details"]["runtime_status"]["ffmpeg_path"]
    assert row["details"]["runtime_resolution_path"]


def test_llama_mtmd_dependency_repair_contract_row_records_native_runtime_resolution(tmp_path):
    from qa.runtime_matrix.rows import llama_mtmd_repair

    row = llama_mtmd_repair.run("llama_mtmd_dependency_repair_contract", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["missing_before"] == ["llama-mtmd-cli or llama-cpp-python Qwen3ASRChatHandler"]
    assert row["details"]["missing_after"] == []
    assert row["details"]["install_kind"] == "native_tool"
    assert row["details"]["repair_action"] == "install_missing"
    assert row["details"]["backend_probe"]["kind"] == "llama_mtmd_runtime_probe"
    assert "ggml.llamacpp" in row["details"]["repair_command"]
    assert row["details"]["runtime_status"]["path"].endswith("llama-mtmd-cli.exe")
    assert row["details"]["runtime_resolution"]["runtime_path"].endswith("llama-mtmd-cli.exe")
    assert row["details"]["runtime_resolution_path"]


def test_directml_provider_conflict_repair_row_records_safe_repair_contract(tmp_path):
    from qa.runtime_matrix.rows import windows_directml_provider

    row = windows_directml_provider.run("directml_provider_conflict_repair", tmp_path, False, False)

    assert row["status"] == "pass"
    assert "onnxruntime conflicts with directml package" in row["details"]["missing_before"]
    assert "onnxruntime DirectML provider" in row["details"]["missing_before"]
    assert row["details"]["missing_after"] == []
    assert row["details"]["repair_result"]["accelerator"] == "directml"
    assert row["details"]["repair_result"]["provider_compatibility_repair"] == "onnxruntime-directml==1.24.4"
    commands = row["details"]["commands"]
    assert any(command[-3:] == ["uninstall", "-y", "onnxruntime"] for command in commands)
    assert any(command[-4:] == ["--upgrade", "--force-reinstall", "--no-deps", "onnxruntime-directml==1.24.4"] for command in commands)


def test_release_manual_matrix_includes_granular_cuda_rows():
    ids = _manual_row_ids()

    assert {
        "nvidia_cuda_hardware_detection",
        "torch_cuda_tensor_smoke",
        "onnxruntime_cuda_tiny_session",
        "faster_whisper_ctranslate2_cuda_smoke",
        "llama_cpp_cuda_smollm_smoke",
        "llama_cpp_vulkan_smollm_smoke",
        "transformers_cuda_unavailable_cpu_fallback",
        "generic_onnx_cuda_unavailable_cpu_fallback",
    } <= ids


def test_runtime_matrix_maps_setup_environment_rows():
    assert ROWS["win11_clean_no_python_setup"].module == "qa.runtime_matrix.rows.setup_environment"
    assert ROWS["windows_sandbox_clean_bootstrap_deploy"].module == "qa.runtime_matrix.rows.clean_vm_bootstrap"
    assert ROWS["windows_sandbox_clean_bootstrap_deploy"].hardware == "clean_windows_vm"
    assert ROWS["clean_vm_zero_dependency_bootstrap"].module == "qa.runtime_matrix.rows.clean_vm_bootstrap"
    assert ROWS["clean_vm_zero_dependency_bootstrap"].hardware == "clean_windows_vm"
    assert ROWS["win10_existing_python_setup"].module == "qa.runtime_matrix.rows.setup_environment"
    assert ROWS["setup_double_click_equivalent"].module == "qa.runtime_matrix.rows.setup_environment"
    assert ROWS["setup_double_click_equivalent"].hardware == "windows"
    assert ROWS["first_run_smoke_json"].module == "qa.runtime_matrix.rows.setup_environment"


def test_clean_vm_bootstrap_row_blocks_without_clean_vm_marker(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import clean_vm_bootstrap

    monkeypatch.delenv(clean_vm_bootstrap.PROOF_ENV, raising=False)
    row = clean_vm_bootstrap.run("clean_vm_zero_dependency_bootstrap", tmp_path, True, True)

    assert row["status"] == "blocked"
    assert clean_vm_bootstrap.PROOF_ENV in row["block_reason"]
    assert "--install-deps --allow-downloads" in row["external_requirement"]
    assert row["details"]["required_sequence"] == [
        "cmd /c setup.bat --doctor --repair-all-safe",
        "cmd /c setup.bat --doctor --repair-model-layouts --allow-downloads",
        "python qa/runtime_matrix/run_row.py --row setup_repair_all_safe --install-deps",
        "python qa/runtime_matrix/run_row.py --row setup_repair_model_layouts --allow-downloads",
        "python qa/runtime_matrix/run_row.py --row same_media_multi_model_smollm_benchmark --install-deps --allow-downloads",
        "python -m app.main --first-run-smoke --json",
    ]


def test_windows_sandbox_deploy_row_generates_launch_bundle(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import clean_vm_bootstrap

    monkeypatch.delenv(clean_vm_bootstrap.SANDBOX_LAUNCH_ENV, raising=False)
    monkeypatch.setattr(clean_vm_bootstrap, "_windows_sandbox_executable", lambda: r"C:\Windows\System32\WindowsSandbox.exe")
    monkeypatch.setattr(clean_vm_bootstrap, "_windows_edition_details", lambda: {"available": True, "edition_id": "Professional", "sandbox_supported_edition": True})
    monkeypatch.setattr(clean_vm_bootstrap, "_sandbox_completion_evidence", lambda: ({"evidence_root": "missing"}, [], ["completion evidence missing"]))
    evidence_dir = ROOT / "Temp" / f"pytest_windows_sandbox_deploy_{tmp_path.name}"
    row = clean_vm_bootstrap.run("windows_sandbox_clean_bootstrap_deploy", evidence_dir, False, False)

    assert row["status"] == "blocked"
    assert clean_vm_bootstrap.SANDBOX_LAUNCH_ENV in row["block_reason"]
    script = Path(row["details"]["bundle"]["script_path"])
    config = Path(row["details"]["bundle"]["config_path"])
    assert script.exists()
    assert config.exists()
    script_text = script.read_text(encoding="utf-8")
    assert "prebootstrap-python-probe.json" in script_text
    assert "EASY_ASR_BENCH_PREBOOTSTRAP_PROBE" in script_text
    assert "$exe = $Command[0]" in script_text
    assert "Invoke-ValidationStep" in script_text
    assert "step-\" + $Name + \".json" in script_text
    assert "setup-dry-run-json" in script_text
    assert "setup-local-no-post-menu" in script_text
    assert '"--local", "--no-post-setup-menu"' in script_text
    assert '$SandboxPython = Join-Path $Repo ".venv\\Scripts\\python.exe"' in script_text
    assert '@($SandboxPython, "qa\\runtime_matrix\\run_row.py"' in script_text
    assert "exit_code_path" in script_text
    assert "Start-Job" in script_text
    assert "Executable not found for sandbox validation step" in script_text
    assert '$commandText = $Command -join "`n"' in script_text
    assert '$StepCommand = @($StepCommandText -split "`n")' in script_text
    assert "clean_vm_zero_dependency_bootstrap" in script_text
    assert '"--validate-real-smoke", "--full-real-smoke", "--allow-downloads"' in script_text
    assert "validate_release_smoke.py" in script_text
    assert "MappedFolder" in config.read_text(encoding="utf-8")


def test_windows_sandbox_deploy_row_blocks_when_feature_missing(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import clean_vm_bootstrap

    monkeypatch.setattr(clean_vm_bootstrap, "_windows_sandbox_executable", lambda: "")
    monkeypatch.setattr(clean_vm_bootstrap, "_windows_edition_details", lambda: {"available": True, "edition_id": "Professional", "sandbox_supported_edition": True})
    monkeypatch.setattr(clean_vm_bootstrap, "_sandbox_completion_evidence", lambda: ({"evidence_root": "missing"}, [], ["completion evidence missing"]))
    row = clean_vm_bootstrap.run("windows_sandbox_clean_bootstrap_deploy", ROOT / "Temp" / f"pytest_windows_sandbox_missing_{tmp_path.name}", False, False)

    assert row["status"] == "blocked"
    assert "WindowsSandbox.exe" in row["block_reason"]
    assert row["details"]["windows_sandbox_executable"] == ""


def test_windows_sandbox_deploy_launch_success_waits_for_completion_evidence(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import clean_vm_bootstrap

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setenv(clean_vm_bootstrap.SANDBOX_LAUNCH_ENV, "1")
    monkeypatch.setattr(clean_vm_bootstrap, "_windows_sandbox_executable", lambda: r"C:\Windows\System32\WindowsSandbox.exe")
    monkeypatch.setattr(clean_vm_bootstrap, "_windows_edition_details", lambda: {"available": True, "edition_id": "Professional", "sandbox_supported_edition": True})
    monkeypatch.setattr(clean_vm_bootstrap, "_sandbox_completion_evidence", lambda: ({"evidence_root": "missing"}, [], ["completion evidence missing"]))
    monkeypatch.setattr(clean_vm_bootstrap.subprocess, "run", lambda *args, **kwargs: Completed())

    row = clean_vm_bootstrap.run("windows_sandbox_clean_bootstrap_deploy", ROOT / "Temp" / f"pytest_windows_sandbox_launch_{tmp_path.name}", False, False)

    assert row["status"] == "blocked"
    assert "completion evidence" in row["block_reason"]
    assert row["details"]["launch_exit_code"] == 0


def test_windows_sandbox_deploy_blocks_when_other_sandbox_running(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import clean_vm_bootstrap

    def fail_run(*args, **kwargs):
        raise AssertionError("sandbox launch should not run while another Sandbox is active")

    monkeypatch.setenv(clean_vm_bootstrap.SANDBOX_LAUNCH_ENV, "1")
    monkeypatch.setattr(clean_vm_bootstrap, "_windows_sandbox_executable", lambda: r"C:\Windows\System32\WindowsSandbox.exe")
    monkeypatch.setattr(clean_vm_bootstrap, "_windows_edition_details", lambda: {"available": True, "edition_id": "Professional", "sandbox_supported_edition": True})
    monkeypatch.setattr(clean_vm_bootstrap, "_sandbox_completion_evidence", lambda: ({"evidence_root": "missing"}, [], ["completion evidence missing"]))
    monkeypatch.setattr(
        clean_vm_bootstrap,
        "_running_windows_sandboxes",
        lambda: [
            {
                "process_id": 123,
                "parent_process_id": 1,
                "name": "WindowsSandbox.exe",
                "command_line": r'"C:\Windows\System32\WindowsSandbox.exe" "E:\other-project\smoke.wsb"',
            }
        ],
    )
    monkeypatch.setattr(clean_vm_bootstrap.subprocess, "run", fail_run)

    row = clean_vm_bootstrap.run("windows_sandbox_clean_bootstrap_deploy", ROOT / "Temp" / f"pytest_windows_sandbox_busy_{tmp_path.name}", False, False)

    assert row["status"] == "blocked"
    assert "another Windows Sandbox" in row["block_reason"]
    assert row["details"]["sandbox_contention"]["other_windows_sandboxes"][0]["process_id"] == 123


def test_windows_sandbox_deploy_row_blocks_on_unsupported_edition(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import clean_vm_bootstrap

    monkeypatch.setattr(clean_vm_bootstrap, "_windows_sandbox_executable", lambda: "")
    monkeypatch.setattr(clean_vm_bootstrap, "_windows_edition_details", lambda: {"available": True, "edition_id": "Core", "sandbox_supported_edition": False})
    monkeypatch.setattr(clean_vm_bootstrap, "_sandbox_completion_evidence", lambda: ({"evidence_root": "missing"}, [], ["completion evidence missing"]))
    row = clean_vm_bootstrap.run("windows_sandbox_clean_bootstrap_deploy", ROOT / "Temp" / f"pytest_windows_sandbox_core_{tmp_path.name}", False, False)

    assert row["status"] == "blocked"
    assert "Core" in row["block_reason"]
    assert "Pro, Enterprise, or Education" in row["external_requirement"]


def test_clean_vm_bootstrap_requires_first_run_repair_evidence():
    source = (ROOT / "qa" / "runtime_matrix" / "rows" / "clean_vm_bootstrap.py").read_text(encoding="utf-8")

    assert 'first_run_payload.get("repair_plan_schema")' in source
    assert "first-run smoke did not include repair-plan evidence" in source
    assert 'first_run_payload.get("repair_command")' in source
    assert "model_layout_repair_command" in source


def test_clean_vm_bootstrap_rejects_incomplete_same_media_evidence():
    from qa.runtime_matrix.rows.clean_vm_bootstrap import _same_media_evidence_failures

    failures = _same_media_evidence_failures(
        {
            "details": {
                "run_adapters": ["faster_whisper"],
                "score_count": 1,
                "dependency_resolution_summary": {
                    "schema": "easy_asr_bench.dependency_resolution_environment.v1",
                    "invalid_resolution_files": 0,
                },
                "dependency_resolution_groups": ["faster_whisper"],
                "last_repair_summary": {"runtime_resolutions": 1},
            }
        }
    )

    assert any("missing adapters" in failure for failure in failures)
    assert any("did not score every required adapter" in failure for failure in failures)
    assert any("missing dependency-resolution groups" in failure for failure in failures)
    assert any("last_repair_summary.cached_runtime_resolutions" in failure for failure in failures)


def test_clean_vm_bootstrap_accepts_complete_same_media_evidence():
    from qa.runtime_matrix.rows.clean_vm_bootstrap import _same_media_evidence_failures

    payload = {
        "details": {
            "run_adapters": [
                "faster_whisper",
                "openai_whisper_pt",
                "generic_onnx_manifest",
                "hf_whisper_asr",
                "hf_transformers_asr",
                "whisper_cpp",
                "gguf_asr_mmproj",
            ],
            "score_count": 7,
            "dependency_resolution_summary": {
                "schema": "easy_asr_bench.dependency_resolution_environment.v1",
                "invalid_resolution_files": 0,
            },
            "dependency_resolution_groups": [
                "python_packaging",
                "media_tools",
                "faster_whisper",
                "onnx",
                "transformers_cpu",
                "whisper_cpp",
                "openai_whisper",
                "llama_cpp",
                "llama_mtmd",
            ],
            "last_repair_summary": {
                "runtime_resolutions": 10,
                "cached_runtime_resolutions": 5,
                "previous_runtime_resolution_valid": 10,
                "previous_runtime_resolution_stale": 0,
            },
        }
    }

    assert _same_media_evidence_failures(payload) == []


def test_clean_vm_bootstrap_rejects_incomplete_setup_repair_evidence():
    from qa.runtime_matrix.rows.clean_vm_bootstrap import _setup_repair_evidence_failures

    failures = _setup_repair_evidence_failures({"details": {"repair_summary": {"runtime_resolutions": 0}}})

    assert "setup_repair_all_safe evidence did not record any runtime resolutions" in failures
    assert any("cached_runtime_resolutions" in failure for failure in failures)
    assert "setup_repair_all_safe evidence missing repair_evidence_path" in failures


def test_clean_vm_bootstrap_rejects_incomplete_model_layout_repair_evidence():
    from qa.runtime_matrix.rows.clean_vm_bootstrap import _model_layout_repair_evidence_failures

    failures = _model_layout_repair_evidence_failures({"details": {"sweep_summary": {"repaired": 0}}})

    assert "setup_repair_model_layouts evidence did not repair any persisted model-layout plan" in failures
    assert any("downloaded_files" in failure for failure in failures)
    assert "setup_repair_model_layouts evidence missing persisted last_execution repair" in failures


def test_setup_environment_rows_emit_concrete_setup_evidence(tmp_path):
    from qa.runtime_matrix.rows import setup_environment

    for row_id in ["win11_clean_no_python_setup", "win10_existing_python_setup", "setup_double_click_equivalent"]:
        row = setup_environment.run(row_id, tmp_path / row_id, False, False)

        assert row["status"] in {"pass", "blocked"}
        assert row["details"]["setup_static_contract"]["missing_markers"] == []
        assert row["details"]["setup_dry_run_local"]["exit_code"] == 0
        assert "python_visible_on_path" in row["details"]["python_probe"]

    double_click = setup_environment.run("setup_double_click_equivalent", tmp_path / "setup_double_click_pass", False, False)
    assert double_click["status"] == "pass"
    assert "block_reason" not in double_click


def test_win11_clean_setup_accepts_prebootstrap_no_python_probe(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import setup_environment

    probe = tmp_path / "prebootstrap.json"
    probe.write_text(
        json.dumps(
            {
                "system": "Windows",
                "release": "11",
                "python_visible_on_path": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(setup_environment.PREBOOTSTRAP_PROBE_ENV, str(probe))
    monkeypatch.setattr(setup_environment.platform, "system", lambda: "Windows")
    monkeypatch.setattr(setup_environment.platform, "release", lambda: "11")
    monkeypatch.setattr(setup_environment.platform, "version", lambda: "10.0.22621")
    monkeypatch.setattr(setup_environment.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(setup_environment, "_python_probe", lambda: {"python_visible_on_path": True})
    monkeypatch.setattr(setup_environment, "_dry_run_local", lambda _evidence_dir: {"exit_code": 0, "stdout_tail": "", "stderr_tail": ""})

    row = setup_environment.run("win11_clean_no_python_setup", tmp_path / "win11", False, False)

    assert row["status"] == "pass"
    assert row["details"]["prebootstrap_probe"]["python_visible_on_path"] is False


def test_win10_existing_python_requires_windows10_build(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import setup_environment

    monkeypatch.setattr(setup_environment.platform, "system", lambda: "Windows")
    monkeypatch.setattr(setup_environment.platform, "release", lambda: "10")
    monkeypatch.setattr(setup_environment.platform, "version", lambda: "10.0.19045")
    monkeypatch.setattr(setup_environment.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(setup_environment, "_python_probe", lambda: {"python_visible_on_path": True})
    monkeypatch.setattr(setup_environment, "_dry_run_local", lambda _evidence_dir: {"exit_code": 0, "stdout_tail": "", "stderr_tail": ""})

    row = setup_environment.run("win10_existing_python_setup", tmp_path / "win10", False, False)

    assert row["status"] == "pass"


def test_win10_existing_python_blocks_on_windows11_build_even_if_release_reports_10(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import setup_environment

    monkeypatch.setattr(setup_environment.platform, "system", lambda: "Windows")
    monkeypatch.setattr(setup_environment.platform, "release", lambda: "10")
    monkeypatch.setattr(setup_environment.platform, "version", lambda: "10.0.22631")
    monkeypatch.setattr(setup_environment.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(setup_environment, "_python_probe", lambda: {"python_visible_on_path": True})
    monkeypatch.setattr(setup_environment, "_dry_run_local", lambda _evidence_dir: {"exit_code": 0, "stdout_tail": "", "stderr_tail": ""})

    row = setup_environment.run("win10_existing_python_setup", tmp_path / "not_win10", False, False)

    assert row["status"] == "blocked"
    assert "version=10.0.22631" in row["block_reason"]


def test_first_run_smoke_json_row_emits_repair_and_action_evidence(tmp_path):
    from qa.runtime_matrix.rows import setup_environment

    row = setup_environment.run("first_run_smoke_json", tmp_path / "first_run_smoke_json", False, False)

    assert row["status"] == "pass"
    payload = row["details"]["payload"]
    assert payload["schema"] == "easy_asr_bench.first_run_smoke.v1"
    assert payload["repair_plan_schema"] == "easy_asr_bench.repair_plan.v1"
    assert payload["repair_command"] == "setup.bat --doctor --repair-all-safe"
    assert payload["model_layout_repair_command"] == "setup.bat --doctor --repair-model-layouts --allow-downloads"
    assert payload["real_smoke_command"] == "setup.bat --doctor --validate-real-smoke"
    assert payload["dead_end"] is False


def test_runtime_matrix_maps_safe_installer_validation_rows():
    assert ROWS["install_path_with_spaces"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["setup_dry_run_verify_release"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["setup_dry_run_json"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["setup_doctor_strict"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["setup_repair_all_safe"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["repair_all_safe_failure_isolation"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["repair_all_safe_stale_cached_resolution"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["repair_plan_issue_classification_contract"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["setup_repair_model_layouts"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["update_preserves_user_data"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["repair_broken_venv"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["uninstall_preserve_user_data"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["destructive_uninstall_requires_phrase"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["setup_verify_release_bad_checksum"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["bad_checksum_fails_before_execution"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["tampered_installer_fails_before_execution"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["interrupted_download_rollback"].module == "qa.runtime_matrix.rows.installer_validation"
    assert ROWS["broken_venv_repair"].module == "qa.runtime_matrix.rows.installer_validation"


def test_install_path_with_spaces_row_runs_non_destructive_dry_run(tmp_path):
    from qa.runtime_matrix.rows import installer_validation

    row = installer_validation.run("install_path_with_spaces", tmp_path, False, False)

    assert row["status"] == "pass"
    assert "Install Path With Spaces" in row["details"]["install_dir"]


def test_tampered_installer_row_fails_before_execution(tmp_path):
    from qa.runtime_matrix.rows import installer_validation

    row = installer_validation.run("tampered_installer_fails_before_execution", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["command"]["exit_code"] != 0
    assert "Integrity check failed before execution" in row["details"]["command"]["stdout_tail"]


def test_setup_verify_release_bad_checksum_row_fails_before_activation(tmp_path):
    from qa.runtime_matrix.rows import installer_validation

    if os.environ.get("GITHUB_ACTIONS"):
        pytest.skip("GitHub Actions PowerShell path transfer fails before this staged bad-checksum row reaches checksum validation; public setup verify still runs as a workflow gate.")
    row = installer_validation.run("setup_verify_release_bad_checksum", tmp_path, False, False)

    assert row["status"] == "pass", row["details"]
    assert row["details"]["command"]["exit_code"] != 0
    output = (row["details"]["command"]["stderr_tail"] + row["details"]["command"]["stdout_tail"]).lower()
    assert ("checksum" in output or "sha256" in output) and "mismatch" in output


def test_setup_dry_run_json_row_emits_parseable_payload(tmp_path):
    from qa.runtime_matrix.rows import installer_validation

    row = installer_validation.run("setup_dry_run_json", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["payload"]["schema"] == "easy_asr_bench.setup_dry_run.v1"
    assert row["details"]["payload"]["mode"] == "dry_run"
    assert row["details"]["payload"]["no_files_modified"] is True


def test_setup_doctor_strict_row_writes_json_evidence(tmp_path):
    from qa.runtime_matrix.rows import installer_validation

    row = installer_validation.run("setup_doctor_strict", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["core_available"] is True
    assert row["details"]["huggingface_cache"]["cache_dir"]
    assert "symlink_supported" in row["details"]["huggingface_cache"]
    artifact_names = {Path(artifact["path"]).name for artifact in row["artifacts"]}
    assert {"config.json", "doctor.json"} <= artifact_names


def test_setup_repair_all_safe_row_writes_backend_probe_evidence(tmp_path):
    from qa.runtime_matrix.rows import installer_validation

    row = installer_validation.run("setup_repair_all_safe", tmp_path, False, False)

    assert row["status"] in {"pass", "blocked"}
    if row["status"] == "blocked":
        assert "--install-deps" in row["external_requirement"]
        assert row["details"]["plan_summary"]["needs_repair"] > 0
        return
    assert row["details"]["plan_summary"]["needs_repair"] == 0
    assert row["details"]["repair_summary"]["backend_probes"] > 0
    assert row["details"]["repair_summary"]["runtime_resolutions"] > 0
    assert "cached_runtime_resolutions" in row["details"]["repair_summary"]
    assert "previous_runtime_resolution_valid" in row["details"]["repair_summary"]
    assert "previous_runtime_resolution_stale" in row["details"]["repair_summary"]
    assert row["details"]["repair_summary"]["backend_probe_failed"] == 0
    assert "accelerator_probe_failed" in row["details"]["repair_summary"]
    artifact_names = {Path(artifact["path"]).name for artifact in row["artifacts"]}
    assert {"config.json", "repair_plan.json", "repair_all_safe.json"} <= artifact_names


def test_setup_repair_all_safe_row_blocks_before_install_without_permission(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import installer_validation

    class Completed:
        stdout = 'diagnostic line before JSON\n{"schema":"easy_asr_bench.repair_plan.v1","summary":{"needs_repair":1,"can_auto_repair":1},"records":[]}'
        stderr = ""
        returncode = 0

    monkeypatch.setattr(
        installer_validation,
        "_run_python_doctor",
        lambda args, evidence_dir: (
            {"command": ["python", "-m", "app.doctor", *args], "exit_code": 0, "stdout_tail": Completed.stdout, "stderr_tail": ""},
            Completed(),
        ),
    )

    row = installer_validation.run("setup_repair_all_safe", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert "--install-deps was not allowed" in row["summary"]
    assert "setup_repair_all_safe --install-deps" in row["external_requirement"]


def test_setup_repair_all_safe_parser_ignores_nested_runtime_schema():
    from qa.runtime_matrix.rows.installer_validation import _last_repair_plan_payload

    text = (
        "diagnostic\n"
        '{"schema":"easy_asr_bench.repair_plan.v1","mode":"repair_all_safe","summary":{"runtime_resolutions":1},'
        '"records":[{"cached_runtime_resolution_check":{"schema":"easy_asr_bench.runtime_resolution_reuse.v1","status":"missing"}}]}\n'
    )

    payload = _last_repair_plan_payload(text, mode="repair_all_safe")

    assert payload["schema"] == "easy_asr_bench.repair_plan.v1"
    assert payload["mode"] == "repair_all_safe"
    assert payload["summary"]["runtime_resolutions"] == 1


def test_repair_all_safe_failure_isolation_row_runs_product_executor(tmp_path):
    from qa.runtime_matrix.rows import installer_validation

    row = installer_validation.run("repair_all_safe_failure_isolation", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["install_order"] == ["onnx", "faster_whisper"]
    assert row["details"]["repair_summary"]["attempted"] == 2
    assert row["details"]["repair_summary"]["failed"] == 1
    assert row["details"]["repair_summary"]["repaired"] == 1
    assert row["details"]["failed_attempt"]["status"] == "failed"
    assert row["details"]["repaired_attempt"]["status"] == "pass"
    artifact_names = {Path(artifact["path"]).name for artifact in row["artifacts"]}
    assert {"repair_all_safe_failure_isolation.json", "repair_all_safe_last.json", "dependency_install_onnx.log"} <= artifact_names


def test_repair_all_safe_stale_cached_resolution_row_refreshes_probe_evidence(tmp_path):
    from qa.runtime_matrix.rows import installer_validation

    row = installer_validation.run("repair_all_safe_stale_cached_resolution", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["cached_runtime_resolution_check"]["status"] == "stale"
    assert "versions" in row["details"]["cached_runtime_resolution_check"]["mismatches"]
    assert row["details"]["probe_calls"] == ["python_packaging"]
    assert row["details"]["install_calls"] == []
    assert row["details"]["previous_runtime_resolution_check"]["status"] == "stale"
    assert row["details"]["refreshed_versions"]["pip"] == "refreshed"
    assert row["details"]["repair_summary"]["cached_runtime_resolutions"] == 0
    assert row["details"]["repair_summary"]["runtime_resolutions"] == 1
    artifact_names = {Path(artifact["path"]).name for artifact in row["artifacts"]}
    assert {"repair_all_safe_stale_cached_resolution.json", "repair_all_safe_last.json", "dependency_resolution_python_packaging.json"} <= artifact_names


def test_repair_plan_issue_classification_contract_row_maps_issue_classes(tmp_path):
    from qa.runtime_matrix.rows import installer_validation

    row = installer_validation.run("repair_plan_issue_classification_contract", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["actions"]["python_packaging"] == "repair_corrupt_install"
    assert row["details"]["actions"]["faster_whisper"] == "replace_incompatible"
    assert row["details"]["actions"]["onnx"] == "replace_conflicting"
    assert row["details"]["actions"]["manual_blocker"] == "block_no_safe_repair"


def test_setup_repair_model_layouts_row_executes_persisted_sidecar_plan(tmp_path):
    from qa.runtime_matrix.rows import installer_validation

    row = installer_validation.run("setup_repair_model_layouts", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["sweep_summary"]["repaired"] == 1
    assert row["details"]["sweep_summary"]["downloaded_files"] == 1
    assert row["details"]["last_execution_summary"]["repaired"] == 1
    artifact_names = {Path(artifact["path"]).name for artifact in row["artifacts"]}
    assert {"config.json", "hf_model_layout_repair_plan.json", "model_layout_repair_sweep.json"} <= artifact_names


def test_update_preservation_row_moves_user_data_into_new_install(tmp_path):
    from qa.runtime_matrix.rows import installer_validation

    row = installer_validation.run("update_preserves_user_data", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["harness"]["preservation_report_schema"] == "easy_asr_bench.install_preservation_report.v1"


def test_interrupted_download_rollback_row_restores_moved_user_data(tmp_path):
    from qa.runtime_matrix.rows import installer_validation

    row = installer_validation.run("interrupted_download_rollback", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["harness"]["phase"] == "after_restore"


def test_broken_venv_repair_row_reports_verified_refresh_without_writes(tmp_path):
    from qa.runtime_matrix.rows import installer_validation

    row = installer_validation.run("broken_venv_repair", tmp_path, False, False)

    assert row["status"] == "pass"
    assert "detected broken venv" in row["details"]["command"]["stdout_tail"]


def test_uninstall_preserve_user_data_row_keeps_user_folders(tmp_path):
    from qa.runtime_matrix.rows import installer_validation

    row = installer_validation.run("uninstall_preserve_user_data", tmp_path, False, False)

    assert row["status"] == "pass"
    assert "User data was preserved" in row["details"]["command"]["stdout_tail"]


def test_destructive_uninstall_requires_phrase_row_preserves_data_without_phrase(tmp_path):
    from qa.runtime_matrix.rows import installer_validation

    row = installer_validation.run("destructive_uninstall_requires_phrase", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["command"]["exit_code"] != 0
    assert "DELETE EASY ASR BENCH USER DATA" in row["details"]["command"]["stdout_tail"]


def test_runtime_matrix_maps_model_folder_scanner_rows():
    assert ROWS["empty_models_folder"].module == "qa.runtime_matrix.rows.model_folder_scanner"
    assert ROWS["empty_models"].module == "qa.runtime_matrix.rows.model_folder_scanner"
    assert ROWS["nested_models_folders"].module == "qa.runtime_matrix.rows.model_folder_scanner"
    assert ROWS["nested_models_scan"].module == "qa.runtime_matrix.rows.model_folder_scanner"


def test_empty_models_runtime_row_scans_to_no_candidates(tmp_path):
    from qa.runtime_matrix.rows import model_folder_scanner

    row = model_folder_scanner.run("empty_models", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["runnable"] == []
    assert row["details"]["unsupported"] == []


def test_nested_models_runtime_row_generates_unique_candidate_ids(tmp_path):
    from qa.runtime_matrix.rows import model_folder_scanner

    row = model_folder_scanner.run("nested_models_scan", tmp_path, False, False)

    assert row["status"] == "pass"
    ids = row["details"]["candidate_ids"]
    assert len(ids) == len(set(ids))
    assert any("group_a" in candidate_id for candidate_id in ids)
    assert any("group_b" in candidate_id for candidate_id in ids)


def test_runtime_matrix_maps_generic_onnx_rows_to_real_tiny_ctc_fixture():
    assert ROWS["generic_onnx_ctc_manifest_v1"].module == "qa.runtime_matrix.rows.generic_onnx_ctc_tiny"
    assert ROWS["generic_onnx_manifest_cpu"].module == "qa.runtime_matrix.rows.generic_onnx_ctc_tiny"
    assert ROWS["generic_onnx_cuda_unavailable_cpu_fallback"].module == "qa.runtime_matrix.rows.generic_onnx_ctc_tiny"
    assert ROWS["generic_onnx_openvino_unavailable_cpu_fallback"].module == "qa.runtime_matrix.rows.generic_onnx_ctc_tiny"
    assert ROWS["generic_onnx_smollm_grading_cpu"].module == "qa.runtime_matrix.rows.generic_onnx_smollm_grading"
    assert ROWS["generic_onnx_smollm_grading_directml"].module == "qa.runtime_matrix.rows.generic_onnx_smollm_grading"
    assert ROWS["generic_onnx_ctc_quality_smollm_grading_cpu"].module == "qa.runtime_matrix.rows.generic_onnx_smollm_grading"
    assert ROWS["real_public_media_generic_onnx_ctc_smollm_grading_cpu"].module == "qa.runtime_matrix.rows.generic_onnx_smollm_grading"
    assert ROWS["real_public_media_generic_onnx_ctc_smollm_grading_cpu"].hardware == "network"
    assert ROWS["real_public_video_generic_onnx_ctc_smollm_grading_cpu"].module == "qa.runtime_matrix.rows.generic_onnx_smollm_grading"
    assert ROWS["real_public_video_generic_onnx_ctc_smollm_grading_cpu"].hardware == "network"
    assert ROWS["generic_onnx_without_manifest_rejected"].module == "qa.runtime_matrix.rows.generic_onnx_ctc_tiny"
    assert ROWS["multi_file_onnx_ar_nar"].module == "qa.runtime_matrix.rows.multi_file_onnx_ar_nar"


def test_required_release_rows_include_generic_onnx_real_video_row():
    data = json.loads((ROOT / "tests" / "fixtures" / "release_required_rows_v2.json").read_text(encoding="utf-8"))

    assert "real_public_video_generic_onnx_ctc_smollm_grading_cpu" in data["rows"]


def test_generic_onnx_without_manifest_row_reports_modelbench_requirement(tmp_path):
    from qa.runtime_matrix.rows import generic_onnx_ctc_tiny

    row = generic_onnx_ctc_tiny.run("generic_onnx_without_manifest_rejected", tmp_path, False, False)

    assert row["status"] == "pass"
    unsupported = row["details"]["unsupported"][0]
    assert unsupported["adapter_name"] == "generic_onnx_manifest"
    assert unsupported["runnable"] is False
    assert "modelbench.json" in unsupported["missing_files"]


def test_generic_onnx_smollm_grading_row_blocks_without_smollm_fixture(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import generic_onnx_smollm_grading

    monkeypatch.setattr(generic_onnx_smollm_grading, "SMOLLM_PATH", tmp_path / "missing-smollm.gguf")

    row = generic_onnx_smollm_grading.run("generic_onnx_smollm_grading_cpu", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert "SmolLM 135M GGUF fixture is not present" in row["summary"]


def test_generic_onnx_quality_smollm_grading_row_blocks_without_smollm_fixture(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import generic_onnx_smollm_grading

    monkeypatch.setattr(generic_onnx_smollm_grading, "SMOLLM_PATH", tmp_path / "missing-smollm.gguf")

    row = generic_onnx_smollm_grading.run("generic_onnx_ctc_quality_smollm_grading_cpu", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert "SmolLM 135M GGUF fixture is not present" in row["summary"]


def test_multi_file_onnx_ar_nar_row_reports_complete_and_incomplete_packages(tmp_path):
    from qa.runtime_matrix.rows import multi_file_onnx_ar_nar

    row = multi_file_onnx_ar_nar.run("multi_file_onnx_ar_nar", tmp_path, False, False)

    assert row["status"] == "pass"
    runnable_adapters = {candidate["adapter_name"] for candidate in row["details"]["runnable"]}
    assert {"granite_onnx_ar", "granite_onnx_nar"} <= runnable_adapters
    missing = {missing for candidate in row["details"]["unsupported"] for missing in candidate["missing_files"]}
    assert {"int8/decode_step.onnx_data", "f32/editor.onnx_data"} <= missing


def test_intel_openvino_row_has_explicit_provider_block_or_pass(tmp_path):
    from qa.runtime_matrix.rows import generic_onnx_ctc_tiny

    row = generic_onnx_ctc_tiny.run("intel_openvino_onnx_smoke", tmp_path, False, False)

    assert row["status"] in {"pass", "blocked"}
    if row["status"] == "blocked":
        assert "OpenVINOExecutionProvider" in row["block_reason"] or "OpenVINOExecutionProvider" in row["external_requirement"]


def test_intel_directml_row_cannot_pass_with_cpu_fallback(tmp_path):
    from qa.runtime_matrix.rows import generic_onnx_ctc_tiny

    row = generic_onnx_ctc_tiny.run("intel_directml_onnx_smoke", tmp_path, False, False)

    assert row["status"] in {"pass", "blocked", "fail"}
    if row["status"] == "pass":
        summary = row["details"]["metrics"]["provider_summary"]
        assert summary["directml_active"] is True
        assert summary["provider_fallback"] is False
    if row["status"] == "blocked":
        assert "DmlExecutionProvider" in row["block_reason"] or "Intel GPU/NPU" in row["block_reason"]
    if row["status"] == "fail":
        assert "fell back" in " ".join(row["details"].get("failures", []))


def test_generic_onnx_openvino_fallback_row_preserves_requested_provider(tmp_path):
    from qa.runtime_matrix.rows import generic_onnx_ctc_tiny

    row = generic_onnx_ctc_tiny.run("generic_onnx_openvino_unavailable_cpu_fallback", tmp_path, False, False)

    assert row["status"] == "pass"
    summary = row["details"]["provider_summary"]
    assert summary["requested_runtime_provider"] == "openvino"
    assert summary["openvino_requested"] is True
    assert row["details"]["transcript"] == "ab"
    if not row["details"]["provider_available"]:
        assert summary["provider_fallback"] is True
        assert summary["active_providers"] == ["CPUExecutionProvider"]


def test_generic_onnx_cuda_fallback_row_preserves_requested_provider(tmp_path):
    from qa.runtime_matrix.rows import generic_onnx_ctc_tiny

    row = generic_onnx_ctc_tiny.run("generic_onnx_cuda_unavailable_cpu_fallback", tmp_path, False, False)

    assert row["status"] == "pass"
    summary = row["details"]["provider_summary"]
    assert summary["requested_runtime_provider"] == "cuda"
    assert summary["cuda_requested"] is True
    assert row["details"]["transcript"] == "ab"
    if not row["details"]["provider_available"]:
        assert summary["provider_fallback"] is True
        assert summary["active_providers"] == ["CPUExecutionProvider"]


def test_cuda_combined_row_has_real_detector(tmp_path):
    from qa.runtime_matrix.rows import cuda_provider_matrix

    row = cuda_provider_matrix.run("nvidia_cuda_torch_onnx_faster_whisper_llama", tmp_path, False, False)

    assert row["status"] in {"pass", "blocked"}
    assert "cuda_provider_checks" in row["details"]
    assert "repair_commands" in row["details"]
    assert "explicit_cuda_requirement_commands" in row["details"]


def test_granular_cuda_rows_emit_pass_or_blocked_evidence(tmp_path):
    from qa.runtime_matrix.rows import cuda_provider_matrix

    for row_id in [
        "nvidia_cuda_hardware_detection",
        "torch_cuda_tensor_smoke",
        "onnxruntime_cuda_tiny_session",
        "faster_whisper_ctranslate2_cuda_smoke",
        "llama_cpp_cuda_smollm_smoke",
    ]:
        row = cuda_provider_matrix.run(row_id, tmp_path / row_id, False, False)

        assert row["status"] in {"pass", "blocked"}
        assert "cuda_provider_checks" in row["details"]
        if row["status"] == "blocked":
            assert row["block_reason"]
            assert row["external_requirement"]


def test_llama_cpp_vulkan_smollm_row_emits_pass_or_blocked_evidence(tmp_path):
    from qa.runtime_matrix.rows import windows_vulkan_runtime

    row = windows_vulkan_runtime.run("llama_cpp_vulkan_smollm_smoke", tmp_path, False, False)

    assert row["status"] in {"pass", "blocked"}
    assert "cuda_provider_checks" in row["details"]
    assert "repair_command" in row["details"]
    assert "llama_cpp_gpu_offload_before" in row["details"]
    if row["status"] == "blocked":
        assert row["block_reason"]
        assert row["external_requirement"]


def test_faster_whisper_cuda_fallback_row_records_runtime_plan(tmp_path):
    from qa.runtime_matrix.rows import cuda_provider_matrix

    row = cuda_provider_matrix.run("faster_whisper_cuda_unavailable_cpu_fallback", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["plan"]["requested_provider"] == "cuda"
    assert row["details"]["plan"]["actual_provider"] in {"cpu", "cuda"}


def test_transformers_cuda_fallback_row_records_runtime_plan_and_repair_command(tmp_path):
    from qa.runtime_matrix.rows import cuda_provider_matrix

    row = cuda_provider_matrix.run("transformers_cuda_unavailable_cpu_fallback", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["plan"]["requested_provider"] == "cuda"
    assert row["details"]["plan"]["actual_provider"] in {"cpu", "cuda"}
    commands = " ".join(row["details"]["explicit_cuda_requirement_commands"]).replace("\\", "/")
    assert "requirements/torch_cuda_cu128.txt" in commands
    assert "requirements/transformers_cpu.txt" in commands
    assert "torch_cuda_available" in row["details"]["hardware"]


def test_openai_whisper_cuda_fallback_row_records_runtime_plan_and_repair_command(tmp_path):
    from qa.runtime_matrix.rows import cuda_provider_matrix

    row = cuda_provider_matrix.run("openai_whisper_cuda_unavailable_cpu_fallback", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["plan"]["requested_provider"] == "cuda"
    assert row["details"]["plan"]["actual_provider"] in {"cpu", "cuda"}
    commands = " ".join(row["details"]["explicit_cuda_requirement_commands"]).replace("\\", "/")
    assert "requirements/torch_cuda_cu128.txt" in commands
    assert "requirements/openai_whisper.txt" in commands


def test_cuda_provider_matrix_llama_command_uses_resolved_wheel(monkeypatch):
    from qa.runtime_matrix.rows.cuda_provider_matrix import _explicit_cuda_requirement_commands

    monkeypatch.setattr("app.dependency_manager.sys.version_info", (3, 12, 10))
    monkeypatch.setattr("app.dependency_manager.nvidia_driver_version", lambda: "555.99")
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: True)

    commands = " ".join(_explicit_cuda_requirement_commands()["llama_cpp"])

    assert "llama-cpp-python/whl/cu125" in commands
    assert "llama_cpp_cuda_cu124.txt" not in commands


def test_runtime_matrix_maps_hf_safetensors_rows_to_real_tiny_fixture_runner():
    assert ROWS["hf_safetensors_asr"].module == "qa.runtime_matrix.rows.hf_safetensors_tiny"
    assert ROWS["hf_whisper_safetensors"].module == "qa.runtime_matrix.rows.hf_safetensors_tiny"
    assert ROWS["hf_whisper_safetensors_cpu"].module == "qa.runtime_matrix.rows.hf_safetensors_tiny"
    assert ROWS["hf_whisper_safetensors_smollm_grading_cpu"].module == "qa.runtime_matrix.rows.hf_safetensors_smollm_grading"
    assert ROWS["hf_whisper_safetensors_quality_smollm_grading_cpu"].module == "qa.runtime_matrix.rows.hf_safetensors_smollm_grading"
    assert ROWS["real_public_media_hf_whisper_safetensors_smollm_grading_cpu"].module == "qa.runtime_matrix.rows.hf_safetensors_smollm_grading"
    assert ROWS["real_public_media_hf_whisper_safetensors_smollm_grading_cpu"].hardware == "network"
    assert ROWS["real_public_video_hf_whisper_safetensors_smollm_grading_cpu"].module == "qa.runtime_matrix.rows.hf_safetensors_smollm_grading"
    assert ROWS["real_public_video_hf_whisper_safetensors_smollm_grading_cpu"].hardware == "network"
    assert ROWS["hf_safetensors_asr_smollm_grading_cpu"].module == "qa.runtime_matrix.rows.hf_safetensors_smollm_grading"
    assert ROWS["hf_safetensors_asr_quality_smollm_grading_cpu"].module == "qa.runtime_matrix.rows.hf_safetensors_smollm_grading"


def test_required_release_rows_include_quality_and_sharded_safetensors_rows():
    data = json.loads((ROOT / "tests" / "fixtures" / "release_required_rows_v2.json").read_text(encoding="utf-8"))

    assert "hf_whisper_safetensors_quality_smollm_grading_cpu" in data["rows"]
    assert "hf_safetensors_asr_quality_smollm_grading_cpu" in data["rows"]
    assert "hf_whisper_sharded_safetensors_smollm_grading_cpu" in data["rows"]


def test_hf_safetensors_smollm_grading_row_blocks_without_smollm_fixture(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import hf_safetensors_smollm_grading

    monkeypatch.setattr(hf_safetensors_smollm_grading, "SMOLLM_PATH", tmp_path / "missing-smollm.gguf")

    row = hf_safetensors_smollm_grading.run("hf_whisper_safetensors_smollm_grading_cpu", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert "SmolLM 135M GGUF fixture is not present" in row["summary"]


def test_hf_whisper_quality_smollm_grading_row_blocks_without_smollm_fixture(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import hf_safetensors_smollm_grading

    monkeypatch.setattr(hf_safetensors_smollm_grading, "SMOLLM_PATH", tmp_path / "missing-smollm.gguf")

    row = hf_safetensors_smollm_grading.run("hf_whisper_safetensors_quality_smollm_grading_cpu", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert "SmolLM 135M GGUF fixture is not present" in row["summary"]


def test_hf_whisper_sharded_safetensors_smollm_grading_row_blocks_without_smollm_fixture(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import hf_safetensors_smollm_grading

    monkeypatch.setattr(hf_safetensors_smollm_grading, "SMOLLM_PATH", tmp_path / "missing-smollm.gguf")

    row = hf_safetensors_smollm_grading.run("hf_whisper_sharded_safetensors_smollm_grading_cpu", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert "SmolLM 135M GGUF fixture is not present" in row["summary"]


def test_hf_safetensors_ctc_quality_smollm_grading_row_blocks_without_smollm_fixture(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import hf_safetensors_smollm_grading

    monkeypatch.setattr(hf_safetensors_smollm_grading, "SMOLLM_PATH", tmp_path / "missing-smollm.gguf")

    row = hf_safetensors_smollm_grading.run("hf_safetensors_asr_quality_smollm_grading_cpu", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert "SmolLM 135M GGUF fixture is not present" in row["summary"]


def test_runtime_matrix_maps_media_rows_to_generated_fixture_runner():
    assert ROWS["wav_mp3_mp4_media"].module == "qa.runtime_matrix.rows.media_fixtures"
    assert ROWS["wav_mp3_mp4_no_audio_corrupt_media"].module == "qa.runtime_matrix.rows.media_fixtures"
    assert ROWS["corrupt_media_readable_error"].module == "qa.runtime_matrix.rows.media_fixtures"
    assert ROWS["no_audio_video_readable_error"].module == "qa.runtime_matrix.rows.media_fixtures"


def test_combined_media_row_records_positive_and_negative_sub_checks(tmp_path):
    from qa.runtime_matrix.rows import media_fixtures

    row = media_fixtures.run("wav_mp3_mp4_no_audio_corrupt_media", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["sub_row_status"] == {
        "wav_mp3_mp4_media": "pass",
        "no_audio_video_readable_error": "pass",
        "corrupt_media_readable_error": "pass",
    }
    assert {"audio_fixtures", "no_audio_video", "corrupt_media"} <= set(row["details"])


def test_runtime_matrix_maps_safetensors_classification_rows():
    assert ROWS["standalone_safetensors_incomplete"].module == "qa.runtime_matrix.rows.safetensors_classification"
    assert ROWS["hf_text_llm_safetensors_unsupported"].module == "qa.runtime_matrix.rows.safetensors_classification"
    assert ROWS["sharded_safetensors_index"].module == "qa.runtime_matrix.rows.safetensors_classification"
    assert ROWS["hf_whisper_sharded_safetensors_smollm_grading_cpu"].module == "qa.runtime_matrix.rows.hf_safetensors_smollm_grading"


def test_runtime_matrix_maps_openai_whisper_pt_safety_rows():
    assert ROWS["openai_whisper_pt_unknown_blocked"].module == "qa.runtime_matrix.rows.openai_whisper_pt_safety"
    assert ROWS["openai_pt_unverified_blocked"].module == "qa.runtime_matrix.rows.openai_whisper_pt_safety"
    assert ROWS["openai_whisper_pt_checksum_verified"].module == "qa.runtime_matrix.rows.openai_whisper_pt_safety"
    assert ROWS["openai_whisper_cuda_unavailable_cpu_fallback"].module == "qa.runtime_matrix.rows.cuda_provider_matrix"


def test_required_release_rows_include_openai_whisper_pt_checksum_runtime_proof():
    data = json.loads((ROOT / "tests" / "fixtures" / "release_required_rows_v2.json").read_text(encoding="utf-8"))

    assert "openai_whisper_pt_checksum_verified" in data["rows"]


def test_runtime_matrix_maps_whisper_cpp_ggml_row():
    assert ROWS["whisper_cpp_ggml"].module == "qa.runtime_matrix.rows.whisper_cpp_ggml"
    assert ROWS["whisper_cpp_ggml_smollm_grading"].module == "qa.runtime_matrix.rows.whisper_cpp_smollm_grading"
    assert ROWS["whisper_cpp_ggml_speech_smollm_grading"].module == "qa.runtime_matrix.rows.whisper_cpp_smollm_grading"
    assert ROWS["real_public_media_whisper_cpp_ggml_smollm_grading"].module == "qa.runtime_matrix.rows.whisper_cpp_smollm_grading"
    assert ROWS["real_public_media_whisper_cpp_ggml_smollm_grading"].hardware == "network"
    assert ROWS["real_public_video_whisper_cpp_ggml_smollm_grading"].module == "qa.runtime_matrix.rows.whisper_cpp_smollm_grading"
    assert ROWS["real_public_video_whisper_cpp_ggml_smollm_grading"].hardware == "network"


def test_required_release_rows_include_quality_bearing_whisper_cpp_speech_row():
    data = json.loads((ROOT / "tests" / "fixtures" / "release_required_rows_v2.json").read_text(encoding="utf-8"))

    assert "whisper_cpp_ggml_speech_smollm_grading" in data["rows"]
    assert "real_public_video_whisper_cpp_ggml_smollm_grading" in data["rows"]


def test_runtime_matrix_maps_real_faster_whisper_smollm_grading_row():
    assert ROWS["faster_whisper_pkg_resources_repair"].module == "qa.runtime_matrix.rows.ctranslate2_dynamic_resolver"
    assert ROWS["faster_whisper_ctranslate2_candidate_fallback_repair"].module == "qa.runtime_matrix.rows.ctranslate2_dynamic_resolver"
    assert ROWS["faster_whisper_vc_runtime_repair"].module == "qa.runtime_matrix.rows.ctranslate2_dynamic_resolver"
    assert ROWS["real_tiny_faster_whisper_smollm_grading"].module == "qa.runtime_matrix.rows.real_faster_whisper_smollm_grading"


def test_transformers_cpu_dependency_repair_row_records_full_import_contract(tmp_path):
    from qa.runtime_matrix.rows import transformers_dependency_repair

    row = transformers_dependency_repair.run("transformers_cpu_dependency_repair_contract", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["repair_action"] == "upgrade_outdated"
    assert row["details"]["missing_after"] == []
    assert row["details"]["runtime_resolution_path"]
    assert set(row["details"]["loaded_modules"]) == {
        "torch",
        "transformers",
        "safetensors",
        "sentencepiece",
        "google.protobuf",
        "torchaudio",
    }
    assert row["details"]["commands"]
    assert "transformers_cpu.txt" in row["details"]["commands"][0]


def test_faster_whisper_pkg_resources_repair_row_records_native_repair_contract(tmp_path):
    from qa.runtime_matrix.rows import ctranslate2_dynamic_resolver

    row = ctranslate2_dynamic_resolver.run("faster_whisper_pkg_resources_repair", tmp_path, False, False)

    assert row["status"] == "pass"
    assert "pkg_resources" in row["details"]["missing_before"]
    assert row["details"]["repair_error"] == ""
    assert row["details"]["commands"]
    command = row["details"]["commands"][0]
    assert "--force-reinstall" in command
    assert command[-2:] == ["-r", row["details"]["expected_requirement"]]
    assert row["details"]["probe_calls"][0]["device"] == "cpu"


def test_faster_whisper_ctranslate2_candidate_fallback_row_records_dynamic_repair_contract(tmp_path):
    from qa.runtime_matrix.rows import ctranslate2_dynamic_resolver

    row = ctranslate2_dynamic_resolver.run("faster_whisper_ctranslate2_candidate_fallback_repair", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["repair_error"] == ""
    assert row["details"]["candidate_install_specs"] == ["ctranslate2==4.8.0", "ctranslate2==4.7.2"]
    assert row["details"]["ignored_out_of_range_versions"] == ["4.10.0", "4.9.0", "4.3.1"]
    assert row["details"]["probe_calls"][-1]["result"] == "pass"


def test_faster_whisper_vc_runtime_repair_row_runs_before_ctranslate2_fallback(tmp_path):
    from qa.runtime_matrix.rows import ctranslate2_dynamic_resolver

    row = ctranslate2_dynamic_resolver.run("faster_whisper_vc_runtime_repair", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["commands"] == [row["details"]["expected_command"]]
    assert row["details"]["expected_command"][:5] == ["winget", "install", "-e", "--id", "Microsoft.VCRedist.2015+.x64"]
    assert row["details"]["probe_calls"][0]["result"] == "pass"
    assert row["details"]["repair_error"] == ""


def test_real_faster_whisper_smollm_grading_row_blocks_without_smollm_fixture(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import real_faster_whisper_smollm_grading

    monkeypatch.setattr(real_faster_whisper_smollm_grading, "SMOLLM_PATH", tmp_path / "missing-smollm.gguf")

    row = real_faster_whisper_smollm_grading.run("real_tiny_faster_whisper_smollm_grading", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert "SmolLM 135M GGUF fixture is not present" in row["summary"]


def test_runtime_matrix_maps_asr_gguf_mmproj_rows():
    assert ROWS["hf_downloader_qwen3_asr_gguf_mmproj_layout"].module == "qa.runtime_matrix.rows.hf_downloader_layouts"
    assert ROWS["hf_downloader_qwen3_asr_gguf_mmproj_public_listing"].module == "qa.runtime_matrix.rows.hf_downloader_layouts"
    assert ROWS["hf_downloader_qwen3_asr_gguf_mmproj_public_listing"].hardware == "network"
    assert ROWS["hf_downloader_qwen3_asr_gguf_mmproj_cached_materialization"].module == "qa.runtime_matrix.rows.hf_downloader_layouts"
    assert ROWS["hf_downloader_qwen3_asr_gguf_mmproj_noninteractive_flow"].module == "qa.runtime_matrix.rows.hf_downloader_layouts"
    assert ROWS["hf_downloader_qwen3_asr_gguf_mmproj_public_noninteractive_flow"].module == "qa.runtime_matrix.rows.hf_downloader_layouts"
    assert ROWS["hf_downloader_qwen3_asr_gguf_mmproj_public_noninteractive_flow"].hardware == "network"
    assert ROWS["hf_downloader_qwen3_asr_gguf_mmproj_public_download_to_asr"].module == "qa.runtime_matrix.rows.hf_downloader_layouts"
    assert ROWS["hf_downloader_qwen3_asr_gguf_mmproj_public_download_to_asr"].hardware == "network"
    assert ROWS["hf_downloader_qwen3_asr_gguf_mmproj_public_real_download_to_asr"].module == "qa.runtime_matrix.rows.hf_downloader_layouts"
    assert ROWS["hf_downloader_qwen3_asr_gguf_mmproj_public_real_download_to_asr"].hardware == "network"
    assert ROWS["hf_downloader_supported_outcome_taxonomy"].module == "qa.runtime_matrix.rows.hf_downloader_layouts"
    assert ROWS["hf_downloader_package_variant_taxonomy"].module == "qa.runtime_matrix.rows.hf_downloader_layouts"
    assert ROWS["audio_asr_gguf_mmproj"].module == "qa.runtime_matrix.rows.gguf_asr_mmproj"
    assert ROWS["gguf_asr_mmproj_pair"].module == "qa.runtime_matrix.rows.gguf_asr_mmproj"
    assert ROWS["real_public_media_gguf_asr_mmproj_smollm_grading"].module == "qa.runtime_matrix.rows.gguf_asr_mmproj"
    assert ROWS["real_public_media_gguf_asr_mmproj_smollm_grading"].hardware == "network"
    assert ROWS["incomplete_audio_asr_gguf_mmproj_rejected"].module == "qa.runtime_matrix.rows.gguf_asr_mmproj"
    assert ROWS["mismatched_audio_asr_gguf_mmproj_rejected"].module == "qa.runtime_matrix.rows.gguf_asr_mmproj"
    assert ROWS["smollm_reference_grading_report"].module == "qa.runtime_matrix.rows.smollm_reference_grading_report"


def test_runtime_matrix_maps_known_unsupported_asr_row():
    assert ROWS["known_unsupported_asr_families_explained"].module == "qa.runtime_matrix.rows.known_unsupported_asr"


def test_known_unsupported_asr_row_explains_each_family(tmp_path):
    from qa.runtime_matrix.rows import known_unsupported_asr

    row = known_unsupported_asr.run("known_unsupported_asr_families_explained", tmp_path, False, False)

    assert row["status"] == "pass"
    formats = {candidate["container_format"] for candidate in row["details"]["unsupported"]}
    assert {"nemo", "funasr", "ort", "mlmodelc", "onnx", "onnx-qwen-asr", "onnx-whisper", "onnx-split-asr"} <= formats
    assert row["details"]["failures"] == []


def test_hf_downloader_supported_outcome_taxonomy_row(tmp_path):
    from qa.runtime_matrix.rows import hf_downloader_layouts

    row = hf_downloader_layouts.run("hf_downloader_supported_outcome_taxonomy", tmp_path, False, False)

    assert row["status"] == "pass"
    cases = row["details"]["cases"]
    assert any(item["adapter_name"] == "faster_whisper" for item in cases["complete_runnable_asr"]["runnable"])
    assert {"config.json", "preprocessor_config.json"} <= set(cases["missing_sidecar_repair"]["files"])
    repair_plan = cases["missing_sidecar_repair"]["structured_repair_plan"]
    assert repair_plan["schema"] == "easy_asr_bench.model_layout_repair_plan.v1"
    assert repair_plan["records"][0]["repair_action"] == "download_exact_missing_files"
    assert set(repair_plan["records"][0]["safe_download_files"]) == {"config.json", "preprocessor_config.json"}
    assert cases["missing_sidecar_repair"]["repair_plan_file_exists"] is True
    execution = cases["missing_sidecar_repair"]["interactive_repair_plan"]["last_execution"]
    assert execution["schema"] == "easy_asr_bench.model_layout_repair_execution.v1"
    assert execution["summary"]["repaired"] == 1
    artifact_names = {Path(artifact["path"]).name for artifact in row["artifacts"]}
    assert "hf_model_layout_repair_plan.json" in artifact_names
    assert any(item["adapter_name"] == "gguf_llm_reference" for item in cases["gguf_reference_llm"]["unsupported"])
    assert cases["unsafe_or_unknown_inspection"]["runnable"] == []


def test_hf_downloader_package_variant_taxonomy_row(tmp_path):
    from qa.runtime_matrix.rows import hf_downloader_layouts

    row = hf_downloader_layouts.run("hf_downloader_package_variant_taxonomy", tmp_path, False, False)

    assert row["status"] == "pass"
    cases = row["details"]["cases"]
    assert cases["sharded_safetensors_index"]["downloaded_names"] == cases["sharded_safetensors_index"]["expected_downloaded_names"]
    assert cases["onnx_variant_isolation"]["variants"] == ["default", "fp16", "q4"]
    assert cases["split_gguf_reference_llm"]["choice"]["task_hint"] == "reference_llm"
    assert row["details"]["failures"] == []


def test_incomplete_asr_gguf_mmproj_row_reports_projector_requirement(tmp_path):
    from qa.runtime_matrix.rows import gguf_asr_mmproj

    row = gguf_asr_mmproj.run("incomplete_audio_asr_gguf_mmproj_rejected", tmp_path, False, False)

    assert row["status"] == "pass"
    candidate = row["details"]["candidates"][0]
    assert candidate["container_format"] == "gguf+mmproj"
    assert "matching mmproj .gguf" in candidate["missing_files"]


def test_mismatched_asr_gguf_mmproj_row_reports_projector_requirement(tmp_path):
    from qa.runtime_matrix.rows import gguf_asr_mmproj

    row = gguf_asr_mmproj.run("mismatched_audio_asr_gguf_mmproj_rejected", tmp_path, False, False)

    assert row["status"] == "pass"
    candidate = row["details"]["candidates"][0]
    assert candidate["container_format"] == "gguf+mmproj"
    assert "matching mmproj .gguf" in candidate["missing_files"]


def test_required_asr_gguf_mmproj_pair_row_targets_public_runtime_fixture(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import gguf_asr_mmproj

    smollm = tmp_path / "SmolLM-135M.Q4_K_M.gguf"
    smollm.write_bytes(b"fixture-smollm")
    monkeypatch.setattr(gguf_asr_mmproj, "SMOLLM_PATH", smollm)
    monkeypatch.setattr(gguf_asr_mmproj, "LOCAL_PUBLIC_FIXTURE_SEARCH_ROOTS", ())

    row = gguf_asr_mmproj.run("gguf_asr_mmproj_pair", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert row["details"]["repo_id"] == "mradermacher/Qwen3-ASR-0.6B-GGUF"
    assert row["details"]["model_file"] == "Qwen3-ASR-0.6B.Q4_K_M.gguf"
    assert row["details"]["mmproj_file"] == "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf"
    assert "--allow-downloads" in row["external_requirement"]


def test_asr_gguf_mmproj_required_row_reuses_cached_public_fixture_without_downloads(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import gguf_asr_mmproj

    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "Temp" / "cached-qwen"
    cache.mkdir(parents=True)
    (cache / gguf_asr_mmproj.PUBLIC_QWEN3_ASR_MODEL_FILE).write_bytes(b"cached-qwen-main")
    (cache / gguf_asr_mmproj.PUBLIC_QWEN3_ASR_MMPROJ_FILE).write_bytes(b"cached-qwen-mmproj")
    evidence_dir = tmp_path / "evidence" / "gguf_asr_mmproj_pair"

    result = gguf_asr_mmproj._ensure_public_fixture("gguf_asr_mmproj_pair", evidence_dir, allow_downloads=False)

    assert result == evidence_dir / "Models" / "mradermacher__Qwen3-ASR-0.6B-GGUF"
    assert (result / gguf_asr_mmproj.PUBLIC_QWEN3_ASR_MODEL_FILE).read_bytes() == b"cached-qwen-main"
    assert (result / gguf_asr_mmproj.PUBLIC_QWEN3_ASR_MMPROJ_FILE).read_bytes() == b"cached-qwen-mmproj"
    manifest = json.loads((result / "model_package.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["main_model"] == gguf_asr_mmproj.PUBLIC_QWEN3_ASR_MODEL_FILE
    assert manifest["artifacts"]["projector"] == gguf_asr_mmproj.PUBLIC_QWEN3_ASR_MMPROJ_FILE


def test_real_public_faster_whisper_row_reuses_cached_model_repo_without_downloads(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import real_public_media_faster_whisper_smollm

    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "Temp" / "Systran__faster-whisper-tiny.en"
    cache.mkdir(parents=True)
    for name in ["model.bin", "config.json", "tokenizer.json"]:
        (cache / name).write_text(name, encoding="utf-8")
    destination_root = tmp_path / "evidence" / "Models"

    result, error = real_public_media_faster_whisper_smollm._ensure_faster_whisper(destination_root, allow_downloads=False)

    assert error is None
    assert result == destination_root / "Systran__faster-whisper-tiny.en"
    assert (result / "model.bin").read_text(encoding="utf-8") == "model.bin"


def test_same_media_model_staging_reuses_cached_quality_fixtures_without_downloads(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import generic_onnx_smollm_grading
    from qa.runtime_matrix.rows import gguf_asr_mmproj
    from qa.runtime_matrix.rows import same_media_multi_model_smollm_benchmark

    monkeypatch.chdir(tmp_path)
    faster = tmp_path / "Temp" / "Systran__faster-whisper-tiny.en"
    faster.mkdir(parents=True)
    for name in ["model.bin", "config.json", "tokenizer.json"]:
        (faster / name).write_text(name, encoding="utf-8")
    hf = tmp_path / "Temp" / "undermachine__wav2vec2-small-finetuned"
    hf.mkdir(parents=True)
    (hf / "config.json").write_text("{}", encoding="utf-8")
    (hf / "model.safetensors").write_bytes(b"safe")
    onnx = tmp_path / "Temp" / "onnx-community__wav2vec2-base-960h-ONNX"
    onnx.mkdir(parents=True)
    (onnx / "model.onnx").write_bytes(b"onnx")
    (onnx / "vocab.json").write_text("{}", encoding="utf-8")
    qwen = tmp_path / "Temp" / "qwen-cache"
    qwen.mkdir(parents=True)
    (qwen / gguf_asr_mmproj.PUBLIC_QWEN3_ASR_MODEL_FILE).write_bytes(b"qwen-main")
    (qwen / gguf_asr_mmproj.PUBLIC_QWEN3_ASR_MMPROJ_FILE).write_bytes(b"qwen-mmproj")
    models_root = tmp_path / "evidence" / "Models"

    faster_result, faster_error = same_media_multi_model_smollm_benchmark._ensure_faster_whisper(models_root, allow_downloads=False)
    hf_result, hf_error = same_media_multi_model_smollm_benchmark._ensure_hf_safetensors(models_root, "undermachine/wav2vec2-small-finetuned", allow_downloads=False)
    onnx_result, onnx_error, onnx_artifacts = same_media_multi_model_smollm_benchmark._ensure_generic_onnx_quality(models_root, allow_downloads=False)
    qwen_result, qwen_error, qwen_artifacts = same_media_multi_model_smollm_benchmark._ensure_gguf_asr_mmproj(models_root, allow_downloads=False)

    assert faster_error is None
    assert faster_result == models_root / "Systran__faster-whisper-tiny.en"
    assert hf_error is None
    assert hf_result == models_root / "undermachine__wav2vec2-small-finetuned"
    assert onnx_error is None
    assert onnx_result == models_root / "onnx-community__wav2vec2-base-960h-ONNX"
    assert {path.name for path in onnx_artifacts} == {"model.onnx", "vocab.json", "modelbench.json"}
    assert json.loads((onnx_result / "modelbench.json").read_text(encoding="utf-8"))["decoding"]["vocab_file"] == "vocab.json"
    assert qwen_error is None
    assert qwen_result == models_root / "mradermacher__Qwen3-ASR-0.6B-GGUF"
    assert {path.name for path in qwen_artifacts} == {
        gguf_asr_mmproj.PUBLIC_QWEN3_ASR_MODEL_FILE,
        gguf_asr_mmproj.PUBLIC_QWEN3_ASR_MMPROJ_FILE,
        "model_package.json",
    }
    assert (qwen_result / gguf_asr_mmproj.PUBLIC_QWEN3_ASR_MODEL_FILE).read_bytes() == b"qwen-main"
    assert generic_onnx_smollm_grading.GENERIC_ONNX_QUALITY_REPO.replace("/", "__") in str(onnx_result)


def test_smollm_reference_grading_row_blocks_without_fixture(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import smollm_reference_grading_report

    monkeypatch.setattr(smollm_reference_grading_report, "SMOLLM_PATH", tmp_path / "missing-smollm.gguf")

    row = smollm_reference_grading_report.run("smollm_reference_grading_report", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert "SmolLM 135M GGUF fixture is not present" in row["summary"]


def test_whisper_cpp_ggml_row_blocks_without_dependency_or_fixture(tmp_path):
    from qa.runtime_matrix.rows import whisper_cpp_ggml

    row = whisper_cpp_ggml.run("whisper_cpp_ggml", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert row["details"]["repo_id"] == "ggerganov/whisper.cpp"
    assert row["details"]["model_file"] == "ggml-tiny.en-q5_1.bin"
    assert row["external_requirement"]


def test_whisper_cpp_smollm_grading_row_blocks_without_smollm_fixture(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import whisper_cpp_smollm_grading

    monkeypatch.setattr(whisper_cpp_smollm_grading, "SMOLLM_PATH", tmp_path / "missing-smollm.gguf")

    row = whisper_cpp_smollm_grading.run("whisper_cpp_ggml_smollm_grading", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert "SmolLM 135M GGUF fixture is not present" in row["summary"]


def test_whisper_cpp_speech_smollm_grading_row_blocks_without_smollm_fixture(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import whisper_cpp_smollm_grading

    monkeypatch.setattr(whisper_cpp_smollm_grading, "SMOLLM_PATH", tmp_path / "missing-smollm.gguf")

    row = whisper_cpp_smollm_grading.run("whisper_cpp_ggml_speech_smollm_grading", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert "SmolLM 135M GGUF fixture is not present" in row["summary"]


def test_openai_whisper_pt_unknown_row_blocks_unsafe_checkpoint(tmp_path):
    from qa.runtime_matrix.rows import openai_whisper_pt_safety

    row = openai_whisper_pt_safety.run("openai_whisper_pt_unknown_blocked", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["model_status"] == "Unsafe blocked"
    assert row["details"]["runnable"] is False
    assert "Blocked .pt checkpoint" in row["details"]["load_guard_error"]


def test_openai_whisper_pt_official_name_wrong_hash_row_blocks_filename_trust(tmp_path):
    from qa.runtime_matrix.rows import openai_whisper_pt_safety

    row = openai_whisper_pt_safety.run("openai_pt_unverified_blocked", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["model_status"] == "Unsafe blocked"
    assert row["details"]["verified_official_checkpoint"] is False
    assert any("filenames are not trusted" in warning for warning in row["details"]["warnings"])


def test_openai_whisper_pt_checksum_verified_row_blocks_without_official_checkpoint(tmp_path):
    from qa.runtime_matrix.rows import openai_whisper_pt_safety

    row = openai_whisper_pt_safety.run("openai_whisper_pt_checksum_verified", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert "official OpenAI Whisper .pt checkpoint" in row["block_reason"]
    assert row["details"]["tiny_pt_sha256"]


def test_runtime_matrix_real_media_manifest_has_required_fixture_kinds():
    data = json.loads((ROOT / "qa" / "runtime_matrix" / "real_media_fixtures.json").read_text(encoding="utf-8"))
    kinds = {fixture["kind"] for fixture in data["fixtures"].values()}

    assert "real_audio_wav" in kinds
    assert "real_video_webm_with_audio" in kinds
    assert "real_video_webm_no_audio" in kinds
    assert "real_video_mp4_with_audio" in kinds
    assert "real_video_mp4_no_audio" in kinds


def test_runtime_matrix_maps_real_media_download_cache_row():
    assert ROWS["real_media_download_cache"].module == "qa.runtime_matrix.rows.real_media_download_cache"
    assert ROWS["real_public_media_faster_whisper_smollm_grading"].module == "qa.runtime_matrix.rows.real_public_media_faster_whisper_smollm"
    assert ROWS["real_public_media_faster_whisper_smollm_grading"].hardware == "network"
    assert ROWS["real_public_video_faster_whisper_smollm_grading"].module == "qa.runtime_matrix.rows.real_public_media_faster_whisper_smollm"
    assert ROWS["real_public_video_faster_whisper_smollm_grading"].hardware == "network"
    assert ROWS["real_public_media_openai_whisper_pt_smollm_grading"].module == "qa.runtime_matrix.rows.real_public_media_faster_whisper_smollm"
    assert ROWS["real_public_media_openai_whisper_pt_smollm_grading"].hardware == "network"
    assert ROWS["real_public_video_openai_whisper_pt_smollm_grading"].module == "qa.runtime_matrix.rows.real_public_media_faster_whisper_smollm"
    assert ROWS["real_public_video_openai_whisper_pt_smollm_grading"].hardware == "network"


def test_real_public_media_faster_whisper_smollm_row_blocks_without_network(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import real_public_media_faster_whisper_smollm

    smollm = tmp_path / "SmolLM-135M.Q4_K_M.gguf"
    smollm.write_bytes(b"fixture")
    monkeypatch.setattr(real_public_media_faster_whisper_smollm, "SMOLLM_PATH", smollm)
    monkeypatch.setattr(real_public_media_faster_whisper_smollm, "_repair_dependencies", lambda *_args: ([], {}, []))
    monkeypatch.setattr(real_public_media_faster_whisper_smollm, "execute_repair_plan", lambda *_args, **_kwargs: {"summary": {}})
    monkeypatch.setattr(real_public_media_faster_whisper_smollm, "_find_cached_file", lambda *_args, **_kwargs: None)

    row = real_public_media_faster_whisper_smollm.run(
        "real_public_media_faster_whisper_smollm_grading",
        tmp_path,
        False,
        False,
    )

    assert row["status"] == "blocked"
    assert "--allow-downloads" in row["external_requirement"]
    assert row["details"]["fixture"]["expected_text"] == "Kabul"

    video_row = real_public_media_faster_whisper_smollm.run(
        "real_public_video_faster_whisper_smollm_grading",
        tmp_path / "video",
        False,
        False,
    )

    assert video_row["status"] == "blocked"
    assert "--allow-downloads" in video_row["external_requirement"]
    assert video_row["details"]["fixture"]["expected_text"] == "You are at risk for HIV."

    openai_pt_row = real_public_media_faster_whisper_smollm.run(
        "real_public_media_openai_whisper_pt_smollm_grading",
        tmp_path / "openai_pt",
        False,
        False,
    )

    assert openai_pt_row["status"] == "blocked"
    assert "--allow-downloads" in openai_pt_row["external_requirement"]
    assert openai_pt_row["details"]["backend"] == "openai_whisper_pt"
    assert openai_pt_row["details"]["fixture"]["expected_text"] == "Kabul"

    openai_pt_video_row = real_public_media_faster_whisper_smollm.run(
        "real_public_video_openai_whisper_pt_smollm_grading",
        tmp_path / "openai_pt_video",
        False,
        False,
    )

    assert openai_pt_video_row["status"] == "blocked"
    assert "--allow-downloads" in openai_pt_video_row["external_requirement"]
    assert openai_pt_video_row["details"]["backend"] == "openai_whisper_pt"
    assert openai_pt_video_row["details"]["fixture"]["expected_text"] == "You are at risk for HIV."


def test_runtime_matrix_maps_same_media_multi_model_row():
    assert ROWS["same_media_multi_model_smollm_benchmark"].module == "qa.runtime_matrix.rows.same_media_multi_model_smollm_benchmark"
    assert ROWS["same_media_multi_model_smollm_benchmark_directml"].module == "qa.runtime_matrix.rows.same_media_multi_model_smollm_benchmark"
    assert ROWS["same_media_multi_model_smollm_benchmark_directml"].hardware == "directml"


def test_dependency_resolution_report_helper_requires_schema_and_repair_counters():
    from qa.runtime_matrix.common import dependency_resolution_report_failures

    results = {
        "environment": {
            "dependency_resolution_environment": {
                "summary": {
                    "schema": "easy_asr_bench.dependency_resolution_environment.v1",
                    "resolution_count": 2,
                    "invalid_resolution_files": 0,
                },
                "resolutions": [
                    {"dependency_group": "onnx"},
                    {"dependency_group": "llama_cpp"},
                ],
                "last_repair_all_safe": {
                    "summary": {
                        "runtime_resolutions": 2,
                        "cached_runtime_resolutions": 1,
                        "previous_runtime_resolution_valid": 2,
                        "previous_runtime_resolution_stale": 0,
                    }
                },
            }
        }
    }

    failures, details = dependency_resolution_report_failures(results, expected_groups={"onnx", "llama_cpp"})

    assert failures == []
    assert details["dependency_resolution_groups"] == ["llama_cpp", "onnx"]
    assert details["last_repair_summary"]["cached_runtime_resolutions"] == 1


def test_dependency_resolution_report_helper_flags_missing_expected_group_when_repair_evidence_exists():
    from qa.runtime_matrix.common import dependency_resolution_report_failures

    results = {
        "environment": {
            "dependency_resolution_environment": {
                "summary": {
                    "schema": "easy_asr_bench.dependency_resolution_environment.v1",
                    "invalid_resolution_files": 0,
                },
                "resolutions": [{"dependency_group": "onnx"}],
                "last_repair_all_safe": {
                    "summary": {
                        "runtime_resolutions": 1,
                        "cached_runtime_resolutions": 0,
                        "previous_runtime_resolution_valid": 1,
                        "previous_runtime_resolution_stale": 0,
                    }
                },
            }
        }
    }

    failures, details = dependency_resolution_report_failures(results, expected_groups={"onnx", "llama_cpp"})

    assert failures == ["dependency resolution report missing groups: llama_cpp"]
    assert details["missing_dependency_resolution_groups"] == ["llama_cpp"]


def test_same_media_multi_model_dependency_groups_include_smollm_and_mtmd():
    from qa.runtime_matrix.rows import same_media_multi_model_smollm_benchmark

    groups = set(same_media_multi_model_smollm_benchmark.GROUPS)

    assert {
        "python_packaging",
        "media_tools",
        "faster_whisper",
        "onnx",
        "transformers_cpu",
        "whisper_cpp",
        "openai_whisper",
        "llama_cpp",
        "llama_mtmd",
    } <= groups


def test_same_media_dependency_repair_recovers_when_post_check_is_clean(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import same_media_multi_model_smollm_benchmark

    monkeypatch.setattr(same_media_multi_model_smollm_benchmark, "GROUPS", ["llama_mtmd"])
    calls = {"missing": 0}

    def fake_missing(group, config):
        calls["missing"] += 1
        return ["llama-mtmd-cli or llama-cpp-python Qwen3ASRChatHandler"] if calls["missing"] == 1 else []

    def fake_install(group, project_root, config, log_path=None):
        if log_path is not None:
            log_path.write_text("fallback repaired runtime\n", encoding="utf-8")
        raise FileNotFoundError("winget")

    monkeypatch.setattr(same_media_multi_model_smollm_benchmark, "missing_modules_for_config", fake_missing)
    monkeypatch.setattr(same_media_multi_model_smollm_benchmark, "install_group_for_config", fake_install)
    monkeypatch.setattr(same_media_multi_model_smollm_benchmark, "recovery_command_for_config", lambda group, config: "repair")

    blockers, details, artifacts = same_media_multi_model_smollm_benchmark._repair_dependencies({}, tmp_path, True)

    assert blockers == []
    assert details["llama_mtmd_repair_recovered_after_exception"] is True
    assert details["llama_mtmd_missing_after"] == []
    assert artifacts == [tmp_path / "llama_mtmd_repair.log"]


def test_same_media_dependency_preflight_retries_safe_repair_before_blocking(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import same_media_multi_model_smollm_benchmark

    calls = []

    def fake_repair(config, evidence_dir, install_deps):
        calls.append(install_deps)
        if not install_deps:
            return ["onnx: missing onnxruntime DirectML provider"], {"onnx_missing_after": ["onnxruntime DirectML provider"]}, []
        return [], {"onnx_missing_after": []}, [tmp_path / "onnx_repair.log"]

    monkeypatch.setattr(same_media_multi_model_smollm_benchmark, "_repair_dependencies", fake_repair)

    blockers, details, artifacts = same_media_multi_model_smollm_benchmark._repair_dependencies_with_safe_retry({}, tmp_path, False)

    assert calls == [False, True]
    assert blockers == []
    assert details["safe_repair_retry_triggered"] is True
    assert details["safe_repair_retry_blockers_before"] == ["onnx: missing onnxruntime DirectML provider"]
    assert details["onnx_missing_after"] == []
    assert details["safe_repair_retry_onnx_missing_after"] == []
    assert artifacts == [tmp_path / "onnx_repair.log"]


def test_folder_batch_uses_safe_dependency_retry():
    from qa.runtime_matrix.rows import real_public_folder_batch_smollm
    from qa.runtime_matrix.rows import same_media_multi_model_smollm_benchmark

    assert real_public_folder_batch_smollm._repair_dependencies_with_safe_retry is same_media_multi_model_smollm_benchmark._repair_dependencies_with_safe_retry


def test_same_media_multi_model_row_blocks_without_smollm_fixture(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import same_media_multi_model_smollm_benchmark

    monkeypatch.setattr(same_media_multi_model_smollm_benchmark, "SMOLLM_PATH", tmp_path / "missing-smollm.gguf")

    row = same_media_multi_model_smollm_benchmark.run("same_media_multi_model_smollm_benchmark", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert "SmolLM 135M GGUF fixture is not present" in row["summary"]


def test_same_media_multi_model_directml_row_blocks_without_smollm_fixture(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import same_media_multi_model_smollm_benchmark

    monkeypatch.setattr(same_media_multi_model_smollm_benchmark, "SMOLLM_PATH", tmp_path / "missing-smollm.gguf")

    row = same_media_multi_model_smollm_benchmark.run("same_media_multi_model_smollm_benchmark_directml", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert "SmolLM 135M GGUF fixture is not present" in row["summary"]


def test_runtime_matrix_maps_report_reference_rows():
    assert ROWS["compare_html_offline"].module == "qa.runtime_matrix.rows.report_reference_validation"
    assert ROWS["compare_html_offline_large_transcript"].module == "qa.runtime_matrix.rows.report_reference_validation"
    assert ROWS["report_atomic_write_failure_cleanup"].module == "qa.runtime_matrix.rows.report_reference_validation"
    assert ROWS["llm_reference_json_import"].module == "qa.runtime_matrix.rows.report_reference_validation"


def test_compare_html_offline_row_writes_scored_report_artifacts(tmp_path):
    from qa.runtime_matrix.rows import report_reference_validation

    row = report_reference_validation.run("compare_html_offline", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["score_status"] == "scored"
    assert row["details"]["scores"]["fixture_fast"]["balanced_rank"] == 1
    artifact_names = {Path(artifact["path"]).name for artifact in row["artifacts"]}
    assert {"results.json", "results.txt", "benchmark.csv", "compare.html", "scored_report.json", "compare_scored.html"} <= artifact_names


def test_compare_html_large_transcript_row_proves_pagination_markers(tmp_path):
    from qa.runtime_matrix.rows import report_reference_validation

    row = report_reference_validation.run("compare_html_offline_large_transcript", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["chunk_count"] == 120


def test_llm_reference_json_import_row_rejects_bad_source_hash(tmp_path):
    from qa.runtime_matrix.rows import report_reference_validation

    row = report_reference_validation.run("llm_reference_json_import", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["score_type"] == "llm_corrected_reference"


def test_report_atomic_write_failure_cleanup_row_removes_partials(tmp_path):
    from qa.runtime_matrix.rows import report_reference_validation

    row = report_reference_validation.run("report_atomic_write_failure_cleanup", tmp_path, False, False)

    assert row["status"] == "pass"
    assert "simulated replace failure" in row["details"]["csv_error"]
    assert "simulated replace failure" in row["details"]["text_error"]
    assert not (Path(row["details"]["output_dir"]) / "benchmark.csv.partial").exists()
    assert not (Path(row["details"]["output_dir"]) / "atomic_text_failure.txt.partial").exists()


def test_watched_folder_queue_contract_row_covers_partial_and_repeat_polls(tmp_path):
    from qa.runtime_matrix.rows import queue_watch_races

    row = queue_watch_races.run("watched_folder_partial_write_queue_contract", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["partial_write"]["partial_bytes"] < row["details"]["partial_write"]["complete_bytes"]
    assert row["details"]["partial_write"]["wait_calls"]
    assert row["details"]["repeat_poll"]["queue_item_count"] == 1
    assert row["details"]["repeat_poll"]["source_count"] == 1
    assert row["details"]["done_fast_key_skip"]["queued"] == []
    assert row["details"]["done_fast_key_skip"]["sha256_file_called"] is False


def test_runtime_matrix_maps_failure_isolation_rows():
    assert ROWS["batch_continues_after_one_model_or_chunk_fails"].module == "qa.runtime_matrix.rows.failure_isolation"
    assert ROWS["one_model_failure_continues"].module == "qa.runtime_matrix.rows.failure_isolation"
    assert ROWS["one_chunk_failure_continues"].module == "qa.runtime_matrix.rows.failure_isolation"
    assert ROWS["dependency_install_declined"].module == "qa.runtime_matrix.rows.failure_isolation"
    assert ROWS["dependency_install_accepted"].module == "qa.runtime_matrix.rows.failure_isolation"
    assert ROWS["watched_folder_partial_write_queue_contract"].module == "qa.runtime_matrix.rows.queue_watch_races"


def test_one_model_failure_row_preserves_successful_report_artifacts(tmp_path):
    from qa.runtime_matrix.rows import failure_isolation

    row = failure_isolation.run("one_model_failure_continues", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["errors_by_model"]["bad_model"] == 1
    artifact_names = {Path(artifact["path"]).name for artifact in row["artifacts"]}
    assert {"results.json", "results.txt", "benchmark.csv", "compare.html"} <= artifact_names


def test_one_chunk_failure_row_normalizes_chunk_error(tmp_path):
    from qa.runtime_matrix.rows import failure_isolation

    row = failure_isolation.run("one_chunk_failure_continues", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["errors_by_model"]["good_model"] == 1


def test_batch_failure_row_covers_model_and_chunk_failure_together(tmp_path):
    from qa.runtime_matrix.rows import failure_isolation

    row = failure_isolation.run("batch_continues_after_one_model_or_chunk_fails", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["run_count"] == 3


def test_dependency_declined_row_skips_only_affected_model(tmp_path):
    from qa.runtime_matrix.rows import failure_isolation

    row = failure_isolation.run("dependency_install_declined", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["kept_model_ids"] == ["good_model"]
    assert row["details"]["skipped_model_ids"] == ["needs_onnx"]


def test_dependency_accepted_row_repairs_and_keeps_affected_model(tmp_path):
    from qa.runtime_matrix.rows import failure_isolation

    row = failure_isolation.run("dependency_install_accepted", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["kept_model_ids"] == ["good_model", "needs_onnx"]
    assert row["details"]["installed_groups"] == ["onnx"]
    assert row["details"]["missing_before"] == {"onnx": ["onnxruntime"]}
    assert row["details"]["missing_after"] == {"onnx": []}
    assert row["details"]["confirmation_prompts"][0]["group"] == "onnx"
    artifact_names = {Path(artifact["path"]).name for artifact in row["artifacts"]}
    assert any(name.startswith("dependency_install_onnx_") for name in artifact_names)


def test_real_media_download_cache_blocks_without_network_permission(tmp_path):
    from qa.runtime_matrix.rows import real_media_download_cache

    row = real_media_download_cache.run("real_media_download_cache", tmp_path, False, False)

    assert row["status"] == "blocked"
    assert row["block_reason"] == "network downloads are disabled for this row"
    assert row["details"]["downloadable_fixture_count"] >= 4
    assert isinstance(row["details"]["source_only_fixtures"], dict)


def test_real_media_download_cache_accepts_public_domain_webm_no_audio_fixture(tmp_path, monkeypatch):
    from qa.runtime_matrix.rows import real_media_download_cache

    def fake_download(_url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fixture public-domain webm bytes")

    def fake_prepare_audio(source: Path, _temp_dir: Path, _config: dict):
        assert source.name == "wikimedia_public_domain_roundhay_webm_no_audio.webm"
        raise RuntimeError(f"No audio stream found in video: {source}")

    monkeypatch.setattr(
        real_media_download_cache,
        "_load_manifest",
        lambda: {
            "fixtures": {
                "wikimedia_public_domain_roundhay_webm_no_audio": {
                    "kind": "real_video_webm_no_audio",
                    "source_page": "https://commons.wikimedia.org/wiki/File:Roundhay_Garden_Scene_(1888).webm",
                    "download_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Roundhay%20Garden%20Scene%20(1888).webm",
                    "license": "Public Domain Mark 1.0 as reported by Wikimedia Commons file page",
                }
            }
        },
    )
    monkeypatch.setattr(real_media_download_cache, "_download", fake_download)
    monkeypatch.setattr(real_media_download_cache, "prepare_audio", fake_prepare_audio)

    row = real_media_download_cache.run("real_media_download_cache", tmp_path, False, True)

    assert row["status"] == "pass"
    downloaded = row["details"]["downloaded"]["wikimedia_public_domain_roundhay_webm_no_audio"]
    assert downloaded["kind"] == "real_video_webm_no_audio"
    assert "No audio stream found in video" in downloaded["no_audio_error"]


def test_real_media_manifest_has_downloadable_wikimedia_audio_fixtures():
    data = json.loads((ROOT / "qa" / "runtime_matrix" / "real_media_fixtures.json").read_text(encoding="utf-8"))

    assert data["fixtures"]["wikimedia_cc0_word_wav"]["download_url"].startswith("https://commons.wikimedia.org/wiki/Special:FilePath/")
    assert data["fixtures"]["wikimedia_public_domain_speech_ogg"]["download_url"].startswith("https://commons.wikimedia.org/wiki/Special:FilePath/")
    roundhay = data["fixtures"]["wikimedia_public_domain_roundhay_webm_no_audio"]
    assert roundhay["kind"] == "real_video_webm_no_audio"
    assert roundhay["download_url"].startswith("https://commons.wikimedia.org/wiki/Special:FilePath/")
    assert "Public Domain" in roundhay["license"]
    assert "no-audio real-video supplement" in roundhay["notes"]


def test_runtime_fixture_manifest_covers_core_runtime_formats():
    data = json.loads((ROOT / "qa" / "runtime_matrix" / "model_fixtures.json").read_text(encoding="utf-8"))
    fixtures = data["fixtures"]
    kinds = {fixture["kind"] for fixture in fixtures.values()}

    assert "faster_whisper_ctranslate2" in kinds
    assert "gguf_reference_llm" in kinds
    assert "hf_whisper_safetensors" in kinds
    assert "hf_whisper_sharded_safetensors_structural" in kinds
    assert "generic_onnx_ctc_manifest" in kinds
    assert "whisper_cpp_ggml" in kinds
    assert "openai_whisper_pt" in kinds
    assert "gguf_asr_mmproj_candidate" in kinds
    assert "same_media_multi_model_benchmark_directml" in kinds
    assert "real_public_audio_fixture" in kinds
    assert "real_public_video_fixture" in kinds
    assert "generic_onnx_ctc_fixture" in fixtures["same_media_multi_model_cpu_set"]["includes"]
    assert "generic_onnx_ctc_fixture" in fixtures["same_media_multi_model_directml_set"]["includes"]
    assert "qwen3_asr_0_6b_gguf" in fixtures["same_media_multi_model_cpu_set"]["includes"]
    assert "qwen3_asr_0_6b_gguf" in fixtures["same_media_multi_model_directml_set"]["includes"]
    assert "hf_whisper_sharded_safetensors_smollm_grading_cpu" in fixtures["hf_tiny_random_whisper_sharded_safetensors"]["rows"]
    assert "llama_cpp_vulkan_smollm_smoke" in fixtures["smollm_135m_gguf"]["rows"]
    assert "real_public_media_faster_whisper_smollm_grading" in fixtures["wikimedia_cc0_word_wav"]["rows"]
    assert "real_public_media_openai_whisper_pt_smollm_grading" in fixtures["wikimedia_cc0_word_wav"]["rows"]
    assert "real_public_media_whisper_cpp_ggml_smollm_grading" in fixtures["wikimedia_cc0_word_wav"]["rows"]
    assert "real_public_media_hf_whisper_safetensors_smollm_grading_cpu" in fixtures["wikimedia_cc0_word_wav"]["rows"]
    assert "real_public_media_generic_onnx_ctc_smollm_grading_cpu" in fixtures["wikimedia_cc0_word_wav"]["rows"]
    assert "real_public_media_gguf_asr_mmproj_smollm_grading" in fixtures["wikimedia_cc0_word_wav"]["rows"]
    assert "real_public_media_openai_whisper_pt_smollm_grading" in fixtures["openai_whisper_tiny_pt"]["rows"]
    assert "real_public_video_openai_whisper_pt_smollm_grading" in fixtures["openai_whisper_tiny_pt"]["rows"]
    assert "real_public_media_whisper_cpp_ggml_smollm_grading" in fixtures["whisper_cpp_tiny_en_q5"]["rows"]
    assert "real_public_video_whisper_cpp_ggml_smollm_grading" in fixtures["whisper_cpp_tiny_en_q5"]["rows"]
    assert "real_public_media_hf_whisper_safetensors_smollm_grading_cpu" in fixtures["hf_whisper_tiny_safetensors"]["rows"]
    assert "real_public_video_hf_whisper_safetensors_smollm_grading_cpu" in fixtures["hf_whisper_tiny_safetensors"]["rows"]
    assert "real_public_media_generic_onnx_ctc_smollm_grading_cpu" in fixtures["generic_onnx_ctc_fixture"]["rows"]
    assert "real_public_video_generic_onnx_ctc_smollm_grading_cpu" in fixtures["generic_onnx_ctc_fixture"]["rows"]
    assert "real_public_media_openai_whisper_pt_smollm_grading" in fixtures["smollm_135m_gguf"]["rows"]
    assert "real_public_video_openai_whisper_pt_smollm_grading" in fixtures["smollm_135m_gguf"]["rows"]
    assert "real_public_media_whisper_cpp_ggml_smollm_grading" in fixtures["smollm_135m_gguf"]["rows"]
    assert "real_public_video_whisper_cpp_ggml_smollm_grading" in fixtures["smollm_135m_gguf"]["rows"]
    assert "real_public_media_hf_whisper_safetensors_smollm_grading_cpu" in fixtures["smollm_135m_gguf"]["rows"]
    assert "real_public_video_hf_whisper_safetensors_smollm_grading_cpu" in fixtures["smollm_135m_gguf"]["rows"]
    assert "real_public_media_generic_onnx_ctc_smollm_grading_cpu" in fixtures["smollm_135m_gguf"]["rows"]
    assert "real_public_video_generic_onnx_ctc_smollm_grading_cpu" in fixtures["smollm_135m_gguf"]["rows"]
    assert "real_public_media_gguf_asr_mmproj_smollm_grading" in fixtures["smollm_135m_gguf"]["rows"]
    assert "real_public_media_gguf_asr_mmproj_smollm_grading" in fixtures["qwen3_asr_0_6b_gguf"]["rows"]
    assert "real_public_video_faster_whisper_smollm_grading" in fixtures["wikimedia_public_domain_spoken_words_webm"]["rows"]
    assert "real_public_video_openai_whisper_pt_smollm_grading" in fixtures["wikimedia_public_domain_spoken_words_webm"]["rows"]
    assert "real_public_video_whisper_cpp_ggml_smollm_grading" in fixtures["wikimedia_public_domain_spoken_words_webm"]["rows"]
    assert "real_public_video_hf_whisper_safetensors_smollm_grading_cpu" in fixtures["wikimedia_public_domain_spoken_words_webm"]["rows"]
    assert "real_public_video_generic_onnx_ctc_smollm_grading_cpu" in fixtures["wikimedia_public_domain_spoken_words_webm"]["rows"]
    fixture_ids = set(fixtures)
    for fixture_id, fixture in fixtures.items():
        for row_id in fixture.get("rows", []):
            assert row_id in ROWS, f"{fixture_id} references unknown runtime row {row_id}"
        for included in fixture.get("includes", []):
            assert included in fixture_ids, f"{fixture_id} includes unknown fixture {included}"


def test_model_fixture_quality_claims_row_separates_structural_and_quality(tmp_path):
    from qa.runtime_matrix.rows import model_fixture_claims

    row = model_fixture_claims.run("model_fixture_quality_claims", tmp_path, False, False)

    assert row["status"] == "pass"
    assert row["details"]["structural_fixture_count"] >= 3
    assert row["details"]["quality_fixture_count"] >= 3
    assert row["details"]["failures"] == []
