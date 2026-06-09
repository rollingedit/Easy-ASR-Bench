import json
import subprocess
import sys
from pathlib import Path

from app.config import DEFAULT_CONFIG
from app.dependency_manager import ACCELERATOR_OVERRIDES, CUDA_INSTALL_OVERRIDES, acceleration_install_decision, cuda_diagnostics, cuda_install_decision, dependency_status, huggingface_cache_status, install_group_for_config, llama_cpp_cuda_tag_for_driver, llama_cpp_gpu_capable, llama_mtmd_cli_status, media_tools_status, missing_modules_for_config, recovery_command, recovery_command_for_config, requirement_version_issues, resolve_llama_cpp_wheel, visual_cpp_redistributable_status
from app.doctor import run_doctor, run_model_layout_repair_sweep, run_real_smoke_validation
from app.main import _dependency_install_confirmation, ensure_dependencies, warn_runtime_dependency_fallbacks
from app.repair_plan import backend_probe_for_group, build_repair_plan, execute_repair_plan, reusable_saved_runtime_resolution
from app.results_writer import dependency_resolution_environment, runtime_environment


def test_dependency_status_includes_descriptions_and_repair_commands():
    status = dependency_status()

    assert status["python_packaging"]["description"]
    missing_packaging = status["python_packaging"]["missing"]
    assert any(item == "pip" or item.startswith("pip<") or item.startswith("pip>") for item in missing_packaging) or status["python_packaging"]["available"]
    assert "requirements/python_packaging.txt" in status["python_packaging"]["recovery_command"]
    assert DEFAULT_CONFIG["runtime"]["prefer_gpu"] is True
    assert DEFAULT_CONFIG["dependency_install"]["allow_cuda_install"] is True
    assert DEFAULT_CONFIG["dependency_install"]["prefer_cpu_safe_defaults"] is False
    assert status["onnx"]["description"]
    assert status["media_tools"]["description"]
    assert status["media_tools"]["requirement_file"] == "requirements/core.txt"
    assert "pip install -r requirements/core.txt" in status["media_tools"]["recovery_command"]
    assert "runtime_status" in status["media_tools"]
    assert status["onnx"]["requirement_file"] == "requirements/onnx.txt"
    assert "pip install -r requirements/onnx.txt" in status["onnx"]["recovery_command"]
    assert "requirements/onnx_cuda.txt" in status["onnx"]["cuda_recovery_command"]
    assert recovery_command("llama_cpp").endswith("requirements/llama_cpp.txt")
    assert "requirements/torch_cuda_cu128.txt" in recovery_command("transformers_cpu", use_cuda=True)
    assert "requirements/faster_whisper_cuda.txt" in recovery_command("faster_whisper", use_cuda=True)
    assert "requirements/llama_cpp_cuda_cu124.txt" in recovery_command("llama_cpp", use_cuda=True)
    assert "torchcodec" not in status["transformers_cpu"]["missing"]
    assert "requirements/openai_whisper.txt" in recovery_command("openai_whisper", use_cuda=True)


def test_dependency_checks_flag_installed_versions_outside_requirement_bounds(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.importlib.metadata.version", lambda package: "4.9.0" if package == "ctranslate2" else "1.2.1")

    issues = requirement_version_issues(["requirements/faster_whisper.txt"])

    assert "ctranslate2<4.9,>=4.4 (installed 4.9.0)" in issues


def test_llama_mtmd_probe_runs_from_cli_directory(monkeypatch, tmp_path: Path):
    from app.dependency_manager import probe_llama_mtmd_cli_path

    cli = tmp_path / "llama-mtmd-cli.exe"
    cli.write_text("", encoding="utf-8")
    captured = {}

    def fake_run(command, cwd=None, text=True, capture_output=True, timeout=15):
        captured["command"] = command
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(command, 0, stdout="Usage: llama-mtmd-cli", stderr="")

    monkeypatch.setattr("app.dependency_manager.subprocess.run", fake_run)

    result = probe_llama_mtmd_cli_path(cli)

    assert result["ok"] is True
    assert captured["command"] == [str(cli), "--help"]
    assert captured["cwd"] == cli.parent


def test_gpu_capable_runtime_groups_have_cuda_overrides():
    expected = {"onnx", "transformers_cpu", "openai_whisper", "faster_whisper", "llama_cpp"}

    assert expected <= set(CUDA_INSTALL_OVERRIDES)
    assert "whisper_cpp" not in CUDA_INSTALL_OVERRIDES
    assert ("onnx", "directml") in ACCELERATOR_OVERRIDES
    assert ("onnx", "openvino") in ACCELERATOR_OVERRIDES
    assert ("llama_cpp", "vulkan") in ACCELERATOR_OVERRIDES


def test_transformers_cpu_group_includes_full_hf_asr_runtime_stack():
    from app.dependency_manager import DEPENDENCY_GROUPS

    modules = set(DEPENDENCY_GROUPS["transformers_cpu"].modules)

    assert {"torch", "transformers", "safetensors", "sentencepiece", "google.protobuf", "torchaudio"} <= modules


def test_python_packaging_group_covers_pkg_resources_repair_prerequisites():
    from app.dependency_manager import DEPENDENCY_GROUPS
    from app.repair_plan import GROUP_IMPORT_PROBES

    modules = set(DEPENDENCY_GROUPS["python_packaging"].modules)

    assert {"pip", "setuptools", "pkg_resources"} <= modules
    assert DEPENDENCY_GROUPS["python_packaging"].requirement_file == "requirements/python_packaging.txt"
    assert set(GROUP_IMPORT_PROBES["python_packaging"]) == {"pip", "setuptools", "pkg_resources"}


def test_media_tools_status_reports_ffmpeg_executable(monkeypatch, tmp_path: Path):
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_text("", encoding="utf-8")
    ffprobe = tmp_path / "ffprobe.exe"
    ffprobe.write_text("", encoding="utf-8")

    monkeypatch.setattr("app.dependency_manager.module_available", lambda module: module == "imageio_ffmpeg")

    class ImageioFFmpeg:
        @staticmethod
        def get_ffmpeg_exe():
            return str(ffmpeg)

    monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", ImageioFFmpeg)

    status = media_tools_status()

    assert status["available"] is True
    assert status["ffmpeg_path"] == str(ffmpeg)
    assert status["ffprobe_available"] is True


def test_media_tools_status_reports_missing_executable(monkeypatch, tmp_path: Path):
    missing = tmp_path / "missing-ffmpeg.exe"
    monkeypatch.setattr("app.dependency_manager.module_available", lambda module: module == "imageio_ffmpeg")

    class ImageioFFmpeg:
        @staticmethod
        def get_ffmpeg_exe():
            return str(missing)

    monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", ImageioFFmpeg)

    status = media_tools_status()

    assert status["available"] is False
    assert status["missing"] == ["ffmpeg executable"]


def test_media_tools_status_reports_missing_imageio_ffmpeg(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.module_available", lambda module: False)

    status = media_tools_status()

    assert status["available"] is False
    assert status["missing"] == ["imageio_ffmpeg"]


def test_llama_cpp_gpu_capable_uses_isolated_marker_probe(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.importlib.util.find_spec", lambda module: object())

    class Completed:
        stdout = "native log\nEASY_ASR_LLAMA_GPU_OFFLOAD=1\n"

    captured = {}

    def fake_run(command, **kwargs):
        captured["code"] = command[2]
        return Completed()

    monkeypatch.setattr("app.dependency_manager.subprocess.run", fake_run)

    assert llama_cpp_gpu_capable() is True
    assert "_dll_handles.append(os.add_dll_directory" in captured["code"]
    assert "_preloaded_dlls.append(ctypes.CDLL" in captured["code"]
    assert "llama_cpp/lib" in captured["code"]


def test_llama_mtmd_group_reports_native_runtime_without_blocking_text_llm(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("app.dependency_manager.llama_cpp_qwen3_asr_handler_available", lambda: False)
    monkeypatch.setattr("app.dependency_manager.shutil.which", lambda name: None)
    monkeypatch.setattr("app.dependency_manager.os.environ", {})

    status = dependency_status({"runtime": {"provider": "cpu"}, "dependency_install": {}, "folders": {"cache": str(tmp_path / "Cache")}})

    assert status["llama_cpp"]["description"] == "GGUF text LLM reference/correction support"
    assert status["llama_mtmd"]["available"] is False
    assert "llama-mtmd-cli or llama-cpp-python Qwen3ASRChatHandler" in status["llama_mtmd"]["missing"]
    assert status["llama_mtmd"]["recovery_command"].startswith("winget install -e --id ggml.llamacpp")
    assert status["llama_mtmd"]["install_kind"] == "native_tool"


def test_llama_mtmd_status_finds_configured_cli(monkeypatch, tmp_path: Path):
    cli = tmp_path / "llama-mtmd-cli.exe"
    cli.write_text("", encoding="utf-8")
    monkeypatch.setattr("app.dependency_manager.llama_cpp_qwen3_asr_handler_available", lambda: False)
    monkeypatch.setattr("app.dependency_manager.shutil.which", lambda name: None)
    monkeypatch.setattr("app.dependency_manager.os.environ", {})
    monkeypatch.setattr("app.dependency_manager.probe_llama_mtmd_cli_path", lambda path: {"ok": True, "path": str(path), "reason": "test"})

    status = llama_mtmd_cli_status({"llama_cpp": {"mtmd_cli_path": str(cli)}})

    assert status["available"] is True
    assert status["path"] == str(cli)
    assert status["source"] == "config"
    assert status["probe"]["ok"] is True


def test_llama_mtmd_status_rejects_inaccessible_cli(monkeypatch, tmp_path: Path):
    cli = tmp_path / "llama-mtmd-cli.exe"
    cli.write_text("", encoding="utf-8")
    monkeypatch.setattr("app.dependency_manager.llama_cpp_qwen3_asr_handler_available", lambda: False)
    monkeypatch.setattr("app.dependency_manager.shutil.which", lambda name: str(cli))
    monkeypatch.setattr("app.dependency_manager.os.environ", {})
    monkeypatch.setattr(
        "app.dependency_manager.probe_llama_mtmd_cli_path",
        lambda path: {"ok": False, "path": str(path), "reason": "permission_denied", "error": "PermissionError: [WinError 5] Access is denied"},
    )

    status = llama_mtmd_cli_status()

    assert status["available"] is False
    assert status["path"] == ""
    assert status["rejected"][0]["reason"] == "permission_denied"


def test_llama_mtmd_status_reuses_verified_runtime_resolution_after_transient_denial(monkeypatch, tmp_path: Path):
    denied_cli = tmp_path / "WinGetPackage" / "llama-mtmd-cli.exe"
    denied_cli.parent.mkdir()
    denied_cli.write_text("", encoding="utf-8")
    cached_cli = tmp_path / "Runtime" / "llama-mtmd-cli.exe"
    cached_cli.parent.mkdir()
    cached_cli.write_text("", encoding="utf-8")
    logs = tmp_path / "Logs"
    logs.mkdir()
    resolution_path = logs / "dependency_resolution_llama_mtmd.json"
    resolution_path.write_text(
        __import__("json").dumps(
            {
                "schema": "easy_asr_bench.runtime_resolution.v1",
                "dependency_group": "llama_mtmd",
                "status": "backend_usable",
                "backend_verified": True,
                "backend_probe_kind": "llama_mtmd_runtime_probe",
                "accelerator_requested": False,
                "versions": {},
                "providers": [],
                "runtime_path": str(cached_cli),
                "config_runtime": {"provider": "auto", "prefer_gpu": True, "fallback_to_cpu": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.dependency_manager.llama_cpp_qwen3_asr_handler_available", lambda: False)
    monkeypatch.setattr("app.dependency_manager.shutil.which", lambda name: str(denied_cli))
    monkeypatch.setattr("app.dependency_manager.os.environ", {})

    def fake_probe(path):
        if Path(path) == denied_cli:
            return {"ok": False, "path": str(path), "reason": "permission_denied", "error": "PermissionError: [WinError 5] Access is denied"}
        return {"ok": True, "path": str(path), "reason": "probe_ok"}

    monkeypatch.setattr("app.dependency_manager.probe_llama_mtmd_cli_path", fake_probe)
    monkeypatch.setattr(
        "app.dependency_manager._stage_llama_mtmd_runtime_copy",
        lambda path, config: {"ok": False, "reason": "stage_copy_failed", "source_path": str(path), "stage_dir": str(tmp_path / "Cache")},
    )

    cache = tmp_path / "Cache"
    status = llama_mtmd_cli_status({"folders": {"logs": str(logs), "cache": str(cache)}})

    assert status["available"] is True
    assert status["path"] != str(denied_cli)
    assert status["source"] in {"cached_runtime_resolution_after_permission_denied", "staged_cached_runtime_resolution_after_permission_denied"}
    assert status["probe"]["source_resolution_path"] == str(resolution_path)
    assert status["rejected"][0]["reason"] == "permission_denied"


def test_llama_mtmd_status_does_not_reuse_same_denied_cached_runtime_without_staged_copy(monkeypatch, tmp_path: Path):
    cli = tmp_path / "WinGetPackage" / "llama-mtmd-cli.exe"
    cli.parent.mkdir()
    cli.write_text("", encoding="utf-8")
    logs = tmp_path / "Logs"
    logs.mkdir()
    resolution_path = logs / "dependency_resolution_llama_mtmd.json"
    resolution_path.write_text(
        __import__("json").dumps(
            {
                "schema": "easy_asr_bench.runtime_resolution.v1",
                "dependency_group": "llama_mtmd",
                "status": "backend_usable",
                "backend_verified": True,
                "backend_probe_kind": "llama_mtmd_runtime_probe",
                "accelerator_requested": False,
                "versions": {},
                "providers": [],
                "runtime_path": str(cli),
                "config_runtime": {"provider": "auto", "prefer_gpu": True, "fallback_to_cpu": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.dependency_manager.llama_cpp_qwen3_asr_handler_available", lambda: False)
    monkeypatch.setattr("app.dependency_manager.shutil.which", lambda name: str(cli))
    monkeypatch.setattr("app.dependency_manager.os.environ", {})
    monkeypatch.setattr(
        "app.dependency_manager.probe_llama_mtmd_cli_path",
        lambda path: {"ok": False, "path": str(path), "reason": "permission_denied", "error": "PermissionError: [WinError 5] Access is denied"},
    )
    monkeypatch.setattr(
        "app.dependency_manager._stage_llama_mtmd_runtime_copy",
        lambda path, config: {"ok": False, "reason": "staged_executable_missing", "source_path": str(path), "stage_dir": str(tmp_path / "Cache")},
    )

    status = llama_mtmd_cli_status({"folders": {"logs": str(logs), "cache": str(tmp_path / "Cache")}})

    assert status["available"] is False
    assert status["path"] == ""
    assert status["rejected"][-1]["reason"] == "cached_runtime_denied_and_staging_failed"
    assert status["cached_runtime_resolution"]["runtime_path"] == str(cli)


def test_llama_mtmd_status_stages_copy_when_installed_cli_is_temporarily_denied(monkeypatch, tmp_path: Path):
    package_dir = tmp_path / "WinGetPackage"
    package_dir.mkdir()
    cli = package_dir / "llama-mtmd-cli.exe"
    cli.write_text("", encoding="utf-8")
    (package_dir / "llama.dll").write_text("", encoding="utf-8")
    cache = tmp_path / "Cache"
    monkeypatch.setattr("app.dependency_manager.llama_cpp_qwen3_asr_handler_available", lambda: False)
    monkeypatch.setattr("app.dependency_manager.shutil.which", lambda name: str(cli))
    monkeypatch.setattr("app.dependency_manager.os.environ", {})

    def fake_probe(path):
        if Path(path) == cli:
            return {"ok": False, "path": str(path), "reason": "permission_denied", "error": "PermissionError: [WinError 5] Access is denied"}
        return {"ok": True, "path": str(path), "reason": "probe_ok"}

    monkeypatch.setattr("app.dependency_manager.probe_llama_mtmd_cli_path", fake_probe)

    status = llama_mtmd_cli_status({"folders": {"cache": str(cache)}})

    assert status["available"] is True
    assert status["source"] == "staged_copy_after_permission_denied"
    assert Path(status["path"]).parent == cache / "native_tools" / "llama_mtmd" / package_dir.name
    assert (Path(status["path"]).parent / "llama.dll").exists()
    assert status["rejected"][0]["reason"] == "permission_denied"


def test_llama_mtmd_status_prefers_staged_copy_after_healthy_probe(monkeypatch, tmp_path: Path):
    package_dir = tmp_path / "WinGetPackage"
    package_dir.mkdir()
    cli = package_dir / "llama-mtmd-cli.exe"
    cli.write_text("", encoding="utf-8")
    (package_dir / "llama.dll").write_text("", encoding="utf-8")
    cache = tmp_path / "Cache"
    monkeypatch.setattr("app.dependency_manager.llama_cpp_qwen3_asr_handler_available", lambda: False)
    monkeypatch.setattr("app.dependency_manager.shutil.which", lambda name: str(cli))
    monkeypatch.setattr("app.dependency_manager.os.environ", {})
    monkeypatch.setattr("app.dependency_manager.probe_llama_mtmd_cli_path", lambda path: {"ok": True, "path": str(path), "reason": "probe_ok"})

    status = llama_mtmd_cli_status({"folders": {"cache": str(cache)}})

    assert status["available"] is True
    assert status["source"] == "staged_copy"
    assert Path(status["path"]).parent == cache / "native_tools" / "llama_mtmd" / package_dir.name
    assert status["source_probe"]["path"] == str(cli)
    assert (Path(status["path"]).parent / "llama.dll").exists()


def test_llama_mtmd_staging_uses_existing_file_when_source_dependency_is_locked(monkeypatch, tmp_path: Path):
    package_dir = tmp_path / "WinGetPackage"
    package_dir.mkdir()
    cli = package_dir / "llama-mtmd-cli.exe"
    cli.write_text("", encoding="utf-8")
    locked = package_dir / "ggml-base.dll"
    locked.write_text("new", encoding="utf-8")
    cache = tmp_path / "Cache"
    staged_dir = cache / "native_tools" / "llama_mtmd" / package_dir.name
    staged_dir.mkdir(parents=True)
    (staged_dir / "ggml-base.dll").write_text("old", encoding="utf-8")
    monkeypatch.setattr("app.dependency_manager.llama_cpp_qwen3_asr_handler_available", lambda: False)
    monkeypatch.setattr("app.dependency_manager.shutil.which", lambda name: str(cli))
    monkeypatch.setattr("app.dependency_manager.os.environ", {})
    monkeypatch.setattr("app.dependency_manager.probe_llama_mtmd_cli_path", lambda path: {"ok": True, "path": str(path), "reason": "probe_ok"})

    real_copy2 = __import__("shutil").copy2

    def fake_copy2(source, destination):
        if Path(source).name == "ggml-base.dll":
            raise PermissionError("locked")
        return real_copy2(source, destination)

    monkeypatch.setattr("app.dependency_manager.shutil.copy2", fake_copy2)

    status = llama_mtmd_cli_status({"folders": {"cache": str(cache)}})

    assert status["available"] is True
    assert status["source"] == "staged_copy"
    assert Path(status["path"]).parent == staged_dir
    assert (staged_dir / "ggml-base.dll").read_text(encoding="utf-8") == "old"
    assert status["staged_runtime"]["copy_failures"] == []


def test_llama_cpp_windows_asset_selector_prefers_vulkan_when_available():
    from app.dependency_manager import _select_llama_cpp_windows_asset

    release = {
        "assets": [
            {"name": "llama-b9000-bin-win-cuda-12.4-x64.zip", "browser_download_url": "https://example/cuda.zip"},
            {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip", "browser_download_url": "https://example/cudart.zip"},
            {"name": "llama-b9000-bin-win-cpu-x64.zip", "browser_download_url": "https://example/cpu.zip"},
            {"name": "llama-b9000-bin-win-vulkan-x64.zip", "browser_download_url": "https://example/vulkan.zip"},
        ]
    }

    assert _select_llama_cpp_windows_asset(release, prefer_vulkan=True)["name"] == "llama-b9000-bin-win-vulkan-x64.zip"
    assert _select_llama_cpp_windows_asset(release, prefer_vulkan=False)["name"] == "llama-b9000-bin-win-cpu-x64.zip"


def test_llama_mtmd_install_falls_back_to_official_portable_release_when_winget_fails(monkeypatch, tmp_path: Path):
    cli = tmp_path / "Cache" / "native_tools" / "llama_mtmd" / "github_b9000" / "llama-mtmd-cli.exe"
    cli.parent.mkdir(parents=True)
    cli.write_text("", encoding="utf-8")
    statuses = iter(
        [
            {"available": False, "missing": ["llama-mtmd-cli"]},
            {"available": False, "missing": ["llama-mtmd-cli"]},
            {"available": True, "path": str(cli), "source": "native_tools_cache", "probe": {"ok": True}},
        ]
    )
    monkeypatch.setattr("app.dependency_manager.llama_cpp_qwen3_asr_handler_available", lambda: False)
    monkeypatch.setattr("app.dependency_manager.llama_mtmd_cli_status", lambda config=None: next(statuses))
    monkeypatch.setattr("app.dependency_manager._run_dependency_command", lambda command, env, log_handle: (_ for _ in ()).throw(OSError("winget logon session failed")))
    monkeypatch.setattr(
        "app.dependency_manager._install_llama_mtmd_portable_release",
        lambda config, log_handle=None: {"ok": True, "path": str(cli), "reason": "portable_release_probe_ok", "probe": {"ok": True}, "asset_name": "llama-b9000-bin-win-cpu-x64.zip"},
    )

    result = install_group_for_config("llama_mtmd", tmp_path, {"folders": {"cache": str(tmp_path / "Cache")}})

    assert result["winget_error"] == "OSError: winget logon session failed"
    assert result["portable_repair"]["ok"] is True
    assert result["post_install_status"]["available"] is True
    assert result["post_install_status"]["path"] == str(cli)


def test_visual_cpp_redistributable_status_reports_non_windows(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.sys.platform", "linux")

    status = visual_cpp_redistributable_status()

    assert status["installed"] is False
    assert status["source"] == "not_windows"
    assert "Microsoft.VCRedist.2015+.x64" in status["repair_command"]


def test_cuda_diagnostics_includes_visual_cpp_status(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.visual_cpp_redistributable_status", lambda: {"installed": True, "version": "14.51.36231", "source": "test"})
    monkeypatch.setattr("app.dependency_manager.module_available", lambda module: False)
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.amd_gpu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.intel_gpu_or_npu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.windows_gpu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.vulkan_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.vulkan_sdk_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.vulkan_build_tooling_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.ctranslate2_cuda_available", lambda: False)

    diagnostics = cuda_diagnostics()

    assert diagnostics["visual_cpp_redistributable"]["installed"] is True
    assert diagnostics["visual_cpp_redistributable"]["version"] == "14.51.36231"


def test_huggingface_cache_status_reports_windows_symlink_note(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("app.dependency_manager.sys.platform", "win32")
    monkeypatch.setattr("app.dependency_manager.module_available", lambda module: module == "huggingface_hub")
    monkeypatch.setattr("app.dependency_manager.os.environ", {"HF_HOME": str(tmp_path / "hf")})
    monkeypatch.setattr("app.dependency_manager._symlink_supported_in_temp", lambda: (False, "OSError: privilege not held"))

    status = huggingface_cache_status()

    assert status["available"] is True
    assert status["cache_dir"] == str(tmp_path / "hf" / "hub")
    assert status["symlink_supported"] is False
    assert status["severity"] == "note"
    assert "cache-space/performance note" in status["messages"][0]
    assert "Developer Mode" in status["repair_guidance"]


def test_repair_plan_records_dependency_group_repair_command(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "app.repair_plan.dependency_status",
        lambda config: {
            "core": {
                "available": True,
                "missing": [],
                "install_kind": "pip",
                "requirement_file": "requirements/core.txt",
                "recovery_command": "python -m pip install -r requirements/core.txt",
            },
            "onnx": {
                "available": False,
                "missing": ["onnxruntime DirectML provider"],
                "install_kind": "pip",
                "requirement_file": "requirements/onnx.txt",
                "recovery_command": "python -m pip install -r requirements/onnx.txt",
                "accelerator_recovery_command": "python -m pip install -r requirements/onnx_directml.txt",
            },
        },
    )
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": group == "onnx", "accelerator": "directml"})

    plan = build_repair_plan({"runtime": {"provider": "directml"}}, project_root=tmp_path)

    assert plan["schema"] == "easy_asr_bench.repair_plan.v1"
    onnx = next(record for record in plan["records"] if record["affected_dependency_group"] == "onnx")
    assert onnx["status"] == "needs_repair"
    assert onnx["can_auto_repair"] is True
    assert onnx["repair_action"] == "install_missing"
    assert "onnx_directml.txt" in onnx["repair_command"]
    assert onnx["log_path"].endswith("dependency_install_onnx.log")
    assert plan["summary"]["needs_repair"] == 1


def test_repair_plan_classifies_update_and_conflict_actions(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "app.repair_plan.dependency_status",
        lambda config: {
            "core": {
                "available": True,
                "missing": [],
                "install_kind": "pip",
                "recovery_command": "python -m pip install -r requirements/core.txt",
            },
            "faster_whisper": {
                "available": False,
                "missing": ["ctranslate2 is outdated enough to fail native load"],
                "install_kind": "pip",
                "recovery_command": "python -m pip install -r requirements/faster_whisper.txt",
            },
            "python_packaging": {
                "available": False,
                "missing": ["pip install is corrupted and import crash prevents bootstrap repair"],
                "install_kind": "pip",
                "recovery_command": "python -m pip install -r requirements/python_packaging.txt",
            },
            "whisper_cpp": {
                "available": False,
                "missing": ["pywhispercpp wheel ABI incompatible with this Python runtime"],
                "install_kind": "pip",
                "recovery_command": "python -m pip install -r requirements/whisper_cpp.txt",
            },
            "onnx": {
                "available": False,
                "missing": ["provider package conflict hides DmlExecutionProvider"],
                "install_kind": "pip",
                "recovery_command": "python -m pip install -r requirements/onnx_directml.txt",
            },
            "manual": {
                "available": False,
                "missing": ["external hardware runtime missing"],
                "install_kind": "manual",
                "recovery_command": "",
            },
        },
    )
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": False, "accelerator": None})

    plan = build_repair_plan({"runtime": {"provider": "auto"}}, project_root=tmp_path)
    actions = {record["affected_dependency_group"]: record["repair_action"] for record in plan["records"]}

    assert actions["core"] == "verify_cached"
    assert actions["faster_whisper"] == "upgrade_outdated"
    assert actions["python_packaging"] == "repair_corrupt_install"
    assert actions["whisper_cpp"] == "replace_incompatible"
    assert actions["onnx"] == "replace_conflicting"
    assert actions["manual"] == "block_no_safe_repair"


def test_bootstrap_repair_uses_structured_repair_all_safe(monkeypatch, tmp_path: Path):
    from app import bootstrap

    calls = []

    def fake_run_doctor(config_path: Path, **kwargs):
        calls.append((config_path, kwargs))
        return 0

    monkeypatch.setattr(bootstrap, "run_doctor", fake_run_doctor)

    assert bootstrap.repair(tmp_path / "config.json") == 0
    assert calls == [(tmp_path / "config.json", {"repair_all_safe": True})]


def _write_model_layout_plan(tmp_path: Path) -> tuple[Path, Path]:
    model_dir = tmp_path / "Models" / "owner__asr__model"
    model_dir.mkdir(parents=True)
    plan_path = model_dir / "hf_model_layout_repair_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "easy_asr_bench.model_layout_repair_plan.v1",
                "repo_id": "owner/asr",
                "revision": None,
                "selected_choice": {
                    "label": "Safetensors",
                    "kind": "safetensors",
                    "task_hint": "metadata_required",
                    "primary_files": ["model.safetensors"],
                    "downloaded_files": ["model.safetensors"],
                },
                "destination": str(model_dir),
                "records": [
                    {
                        "issue_id": "model_layout:incomplete",
                        "repair_action": "download_exact_missing_files",
                        "safe_download_files": ["config.json"],
                        "can_auto_repair": True,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "folders": {
                    "models": str(tmp_path / "Models"),
                    "input": str(tmp_path / "Input"),
                    "output": str(tmp_path / "Output"),
                    "temp": str(tmp_path / "Temp"),
                    "logs": str(tmp_path / "Logs"),
                    "cache": str(tmp_path / "Cache"),
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path, plan_path


def test_doctor_model_layout_repair_sweep_blocks_without_download_permission(tmp_path: Path):
    config_path, plan_path = _write_model_layout_plan(tmp_path)

    report = run_model_layout_repair_sweep(config_path, allow_downloads=False)

    assert report["schema"] == "easy_asr_bench.model_layout_repair_sweep.v1"
    assert report["summary"]["blocked"] == 1
    assert report["executions"][0]["plan_path"] == str(plan_path)
    assert "not approved" in report["executions"][0]["records"][0]["block_reason"]


def test_doctor_model_layout_repair_sweep_executes_approved_plan(monkeypatch, tmp_path: Path):
    config_path, plan_path = _write_model_layout_plan(tmp_path)
    downloaded: list[str] = []

    def fake_download(repo_id: str, revision: str | None, filename: str, destination: Path, relative_name: str | None = None) -> Path:
        downloaded.append(filename)
        path = destination / (relative_name or filename)
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr("app.hf_model_downloader.list_repo_files", lambda ref: ["model.safetensors", "config.json"])
    monkeypatch.setattr("app.hf_model_downloader._download_file", fake_download)

    report = run_model_layout_repair_sweep(config_path, allow_downloads=True)

    persisted = json.loads(plan_path.read_text(encoding="utf-8"))
    assert report["summary"]["repaired"] == 1
    assert report["summary"]["downloaded_files"] == 1
    assert downloaded == ["config.json"]
    assert persisted["last_execution"]["summary"]["repaired"] == 1


def test_execute_repair_plan_repairs_group_and_records_after_state(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "app.repair_plan.dependency_status",
        lambda config: {
            "onnx": {
                "available": False,
                "missing": ["onnxruntime"],
                "install_kind": "pip",
                "requirement_file": "requirements/onnx.txt",
                "recovery_command": "python -m pip install -r requirements/onnx.txt",
            },
        },
    )
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": False, "accelerator": None})
    calls = []

    def fake_install(group, project_root, config, log_path=None):
        calls.append((group, project_root, log_path))
        return {"installed_group": group}

    monkeypatch.setattr("app.repair_plan.install_group_for_config", fake_install)
    monkeypatch.setattr("app.repair_plan.missing_modules_for_config", lambda group, config: [])
    monkeypatch.setattr("app.repair_plan.backend_probe_for_group", lambda group, config: {"kind": "test_probe", "ok": True, "group": group})

    plan = execute_repair_plan({"runtime": {"provider": "cpu"}}, project_root=tmp_path)

    onnx = plan["records"][0]
    assert plan["mode"] == "repair_all_safe"
    assert onnx["status"] == "repaired"
    assert onnx["attempts"][0]["status"] == "pass"
    assert onnx["attempts"][0]["repair_action"] == "install_missing"
    assert onnx["attempts"][0]["install_decision"] == {"installed_group": "onnx"}
    assert onnx["after"]["available"] is True
    assert onnx["after"]["missing"] == []
    assert onnx["after"]["repair_action"] == "install_missing"
    assert onnx["after"]["backend_probe"] == {"kind": "test_probe", "ok": True, "group": "onnx"}
    assert onnx["after"]["runtime_resolution"]["schema"] == "easy_asr_bench.runtime_resolution.v1"
    assert onnx["after"]["runtime_resolution"]["dependency_group"] == "onnx"
    assert onnx["after"]["runtime_resolution"]["status"] == "backend_usable"
    assert Path(onnx["after"]["runtime_resolution_path"]).name == "dependency_resolution_onnx.json"
    persisted = __import__("json").loads(Path(onnx["after"]["runtime_resolution_path"]).read_text(encoding="utf-8"))
    assert persisted["dependency_group"] == "onnx"
    assert persisted["backend_verified"] is True
    assert onnx["after"]["previous_runtime_resolution_check"]["status"] == "missing"
    assert plan["summary"]["repaired"] == 1
    assert plan["summary"]["backend_probes"] == 1
    assert plan["summary"]["runtime_resolutions"] == 1
    assert plan["summary"]["previous_runtime_resolution_valid"] == 0
    assert plan["summary"]["previous_runtime_resolution_stale"] == 0
    assert plan["summary"]["backend_probe_failed"] == 0
    assert plan["summary"]["accelerator_probes"] == 0
    assert plan["summary"]["accelerator_probe_failed"] == 0
    evidence_path = tmp_path / "Logs" / "repair_all_safe_last.json"
    assert evidence_path.exists()
    evidence = __import__("json").loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["mode"] == "repair_all_safe"
    assert evidence["summary"]["repair_evidence_path"] == str(evidence_path)
    assert evidence["records"][0]["after"]["runtime_resolution_path"] == str(tmp_path / "Logs" / "dependency_resolution_onnx.json")
    assert calls[0][0] == "onnx"
    assert str(calls[0][2]).endswith("dependency_install_onnx.log")


def test_execute_repair_plan_failure_isolation_keeps_next_group(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "app.repair_plan.dependency_status",
        lambda config: {
            "onnx": {
                "available": False,
                "missing": ["onnxruntime"],
                "install_kind": "pip",
                "requirement_file": "requirements/onnx.txt",
                "recovery_command": "python -m pip install -r requirements/onnx.txt",
            },
            "faster_whisper": {
                "available": False,
                "missing": ["ctranslate2"],
                "install_kind": "pip",
                "requirement_file": "requirements/faster_whisper.txt",
                "recovery_command": "python -m pip install -r requirements/faster_whisper.txt",
            },
        },
    )
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": False, "accelerator": None})
    calls = []

    def fake_install(group, project_root, config, log_path=None):
        calls.append(group)
        if group == "onnx":
            raise RuntimeError("pip failed")
        return {"installed_group": group}

    monkeypatch.setattr("app.repair_plan.install_group_for_config", fake_install)
    monkeypatch.setattr("app.repair_plan.missing_modules_for_config", lambda group, config: [])
    monkeypatch.setattr("app.repair_plan.backend_probe_for_group", lambda group, config: {"kind": "test_probe", "ok": True, "group": group})

    plan = execute_repair_plan({"runtime": {"provider": "cpu"}}, project_root=tmp_path)

    by_group = {record["affected_dependency_group"]: record for record in plan["records"]}
    assert calls == ["onnx", "faster_whisper"]
    assert by_group["onnx"]["status"] == "repair_failed"
    assert by_group["onnx"]["after"]["repair_action"] == "install_missing"
    assert by_group["onnx"]["attempts"][0]["message"] == "pip failed"
    assert by_group["faster_whisper"]["status"] == "repaired"
    assert by_group["faster_whisper"]["attempts"][0]["repair_action"] == "install_missing"
    assert by_group["faster_whisper"]["after"]["backend_probe"]["group"] == "faster_whisper"
    assert by_group["faster_whisper"]["after"]["runtime_resolution"]["dependency_group"] == "faster_whisper"
    assert "runtime_resolution_path" not in by_group["onnx"]["after"]
    assert by_group["faster_whisper"]["after"]["previous_runtime_resolution_check"]["status"] == "missing"
    assert plan["summary"]["failed"] == 1
    assert plan["summary"]["repaired"] == 1
    assert plan["summary"]["backend_probes"] == 1
    assert plan["summary"]["runtime_resolutions"] == 1
    assert plan["summary"]["backend_probe_failed"] == 0
    assert plan["summary"]["accelerator_probes"] == 0
    assert plan["summary"]["accelerator_probe_failed"] == 0


def test_execute_repair_plan_reuses_valid_cached_runtime_resolution(monkeypatch, tmp_path: Path):
    resolution_path = tmp_path / "Logs" / "dependency_resolution_faster_whisper.json"
    resolution_path.parent.mkdir(parents=True)
    resolution_path.write_text(
        __import__("json").dumps(
            {
                "schema": "easy_asr_bench.runtime_resolution.v1",
                "dependency_group": "faster_whisper",
                "status": "backend_usable",
                "backend_verified": True,
                "backend_probe_kind": "faster_whisper_ctranslate2_import_probe",
                "accelerator_requested": False,
                "accelerator": "",
                "accelerator_verified": False,
                "versions": {"faster_whisper": "1.2.1", "ctranslate2": "4.8.0"},
                "providers": [],
                "runtime_path": "",
                "config_runtime": {"provider": "cpu", "prefer_gpu": False, "fallback_to_cpu": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.repair_plan.dependency_status",
        lambda config: {
            "faster_whisper": {
                "available": True,
                "missing": [],
                "install_kind": "pip",
                "requirement_file": "requirements/faster_whisper.txt",
                "recovery_command": "python -m pip install -r requirements/faster_whisper.txt",
            },
        },
    )
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": False, "accelerator": None})
    monkeypatch.setattr("app.repair_plan.missing_modules_for_config", lambda group, config: [])
    monkeypatch.setattr("app.repair_plan.importlib.metadata.version", lambda package: {"faster_whisper": "1.2.1", "ctranslate2": "4.8.0"}[package])
    monkeypatch.setattr("app.repair_plan.backend_probe_for_group", lambda group, config: (_ for _ in ()).throw(AssertionError("cached resolution should skip backend probe")))

    plan = execute_repair_plan({"runtime": {"provider": "cpu", "prefer_gpu": False, "fallback_to_cpu": True}}, project_root=tmp_path)

    record = plan["records"][0]
    assert record["after"]["repair_result"] == "already_ok_cached_resolution"
    assert record["after"]["backend_probe"]["kind"] == "cached_runtime_resolution"
    assert record["after"]["runtime_resolution_path"] == str(resolution_path)
    assert record["cached_runtime_resolution_check"]["status"] == "valid"
    assert plan["summary"]["cached_runtime_resolutions"] == 1
    assert plan["summary"]["backend_probes"] == 1


def test_reusable_runtime_resolution_marks_inaccessible_native_path_stale(monkeypatch, tmp_path: Path):
    runtime_path = tmp_path / "Cache" / "native_tools" / "llama_mtmd" / "package" / "llama-mtmd-cli.exe"
    resolution_path = tmp_path / "Logs" / "dependency_resolution_llama_mtmd.json"
    resolution_path.parent.mkdir(parents=True)
    resolution_path.write_text(
        __import__("json").dumps(
            {
                "schema": "easy_asr_bench.runtime_resolution.v1",
                "dependency_group": "llama_mtmd",
                "status": "backend_usable",
                "backend_verified": True,
                "backend_probe_kind": "llama_mtmd_runtime_probe",
                "accelerator_requested": False,
                "accelerator": "",
                "accelerator_verified": False,
                "versions": {},
                "providers": [],
                "runtime_path": str(runtime_path),
                "config_runtime": {"provider": "cpu", "prefer_gpu": False, "fallback_to_cpu": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.repair_plan._runtime_path_access_check",
        lambda _path: {"exists": False, "error_type": "PermissionError", "error": "[WinError 5] Access is denied"},
    )
    monkeypatch.setattr("app.repair_plan.missing_modules_for_config", lambda group, config: [])

    check = reusable_saved_runtime_resolution(
        "llama_mtmd",
        {
            "runtime": {"provider": "cpu", "prefer_gpu": False, "fallback_to_cpu": True},
            "folders": {"cache": str(tmp_path / "Cache")},
        },
        tmp_path,
    )

    assert check["status"] == "stale"
    assert "runtime_path_inaccessible" in check["mismatches"]
    assert "runtime_path" in check["mismatches"]
    assert check["runtime_path_access"]["error_type"] == "PermissionError"


def test_execute_repair_plan_reprobes_stale_cached_runtime_resolution(monkeypatch, tmp_path: Path):
    resolution_path = tmp_path / "Logs" / "dependency_resolution_faster_whisper.json"
    resolution_path.parent.mkdir(parents=True)
    resolution_path.write_text(
        __import__("json").dumps(
            {
                "schema": "easy_asr_bench.runtime_resolution.v1",
                "dependency_group": "faster_whisper",
                "status": "backend_usable",
                "backend_verified": True,
                "backend_probe_kind": "faster_whisper_ctranslate2_import_probe",
                "accelerator_requested": False,
                "versions": {"faster_whisper": "1.2.1", "ctranslate2": "4.7.2"},
                "providers": [],
                "runtime_path": "",
                "config_runtime": {"provider": "cpu", "prefer_gpu": False, "fallback_to_cpu": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.repair_plan.dependency_status",
        lambda config: {
            "faster_whisper": {
                "available": True,
                "missing": [],
                "install_kind": "pip",
                "requirement_file": "requirements/faster_whisper.txt",
                "recovery_command": "python -m pip install -r requirements/faster_whisper.txt",
            },
        },
    )
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": False, "accelerator": None})
    monkeypatch.setattr("app.repair_plan.missing_modules_for_config", lambda group, config: [])
    monkeypatch.setattr("app.repair_plan.importlib.metadata.version", lambda package: {"faster_whisper": "1.2.1", "ctranslate2": "4.8.0"}[package])
    monkeypatch.setattr("app.repair_plan.backend_probe_for_group", lambda group, config: {"kind": "test_probe", "ok": True, "group": group})

    plan = execute_repair_plan({"runtime": {"provider": "cpu", "prefer_gpu": False, "fallback_to_cpu": True}}, project_root=tmp_path)

    record = plan["records"][0]
    assert record["cached_runtime_resolution_check"]["status"] == "stale"
    assert record["cached_runtime_resolution_check"]["mismatches"] == ["versions"]
    assert record["after"]["repair_result"] == "already_ok"
    assert record["after"]["backend_probe"]["kind"] == "test_probe"
    assert record["after"]["previous_runtime_resolution_check"]["status"] == "stale"
    assert plan["summary"]["cached_runtime_resolutions"] == 0


def test_execute_repair_plan_reuses_valid_onnx_provider_resolution(monkeypatch, tmp_path: Path):
    resolution_path = tmp_path / "Logs" / "dependency_resolution_onnx.json"
    resolution_path.parent.mkdir(parents=True)
    resolution_path.write_text(
        __import__("json").dumps(
            {
                "schema": "easy_asr_bench.runtime_resolution.v1",
                "dependency_group": "onnx",
                "status": "backend_usable",
                "backend_verified": True,
                "backend_probe_kind": "onnxruntime_provider_probe",
                "accelerator_requested": False,
                "versions": {},
                "providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
                "runtime_path": "",
                "config_runtime": {"provider": "auto", "prefer_gpu": True, "fallback_to_cpu": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.repair_plan.dependency_status",
        lambda config: {
            "onnx": {
                "available": True,
                "missing": [],
                "install_kind": "pip",
                "requirement_file": "requirements/onnx.txt",
                "recovery_command": "python -m pip install -r requirements/onnx.txt",
            },
        },
    )
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": False, "accelerator": None})
    monkeypatch.setattr("app.repair_plan.missing_modules_for_config", lambda group, config: [])
    monkeypatch.setattr("app.dependency_manager.onnxruntime_available_providers", lambda: (["DmlExecutionProvider", "CPUExecutionProvider"], ""))
    monkeypatch.setattr("app.repair_plan.backend_probe_for_group", lambda group, config: (_ for _ in ()).throw(AssertionError("cached ONNX resolution should skip backend probe")))

    plan = execute_repair_plan({"runtime": {"provider": "auto", "prefer_gpu": True, "fallback_to_cpu": True}}, project_root=tmp_path)

    record = plan["records"][0]
    assert record["after"]["repair_result"] == "already_ok_cached_resolution"
    assert record["after"]["backend_probe"]["cached_backend_probe_kind"] == "onnxruntime_provider_probe"
    assert record["cached_runtime_resolution_check"]["status"] == "valid"
    assert plan["summary"]["cached_runtime_resolutions"] == 1


def test_execute_repair_plan_reuses_verified_onnx_accelerator_resolution(monkeypatch, tmp_path: Path):
    resolution_path = tmp_path / "Logs" / "dependency_resolution_onnx.json"
    resolution_path.parent.mkdir(parents=True)
    resolution_path.write_text(
        __import__("json").dumps(
            {
                "schema": "easy_asr_bench.runtime_resolution.v1",
                "dependency_group": "onnx",
                "status": "accelerator_verified",
                "backend_verified": True,
                "backend_probe_kind": "onnxruntime_provider_probe",
                "accelerator_requested": True,
                "accelerator": "directml",
                "accelerator_verified": True,
                "versions": {},
                "providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
                "runtime_path": "",
                "config_runtime": {"provider": "directml", "prefer_gpu": True, "fallback_to_cpu": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.repair_plan.dependency_status",
        lambda config: {
            "onnx": {
                "available": True,
                "missing": [],
                "install_kind": "pip",
                "requirement_file": "requirements/onnx_directml.txt",
                "recovery_command": "python -m pip install -r requirements/onnx_directml.txt",
            },
        },
    )
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": True, "accelerator": "directml"})
    monkeypatch.setattr("app.repair_plan.missing_modules_for_config", lambda group, config: [])
    monkeypatch.setattr("app.dependency_manager.onnxruntime_available_providers", lambda: (["DmlExecutionProvider", "CPUExecutionProvider"], ""))
    monkeypatch.setattr("app.repair_plan.backend_probe_for_group", lambda group, config: (_ for _ in ()).throw(AssertionError("verified ONNX accelerator resolution should skip backend probe")))

    plan = execute_repair_plan({"runtime": {"provider": "directml", "prefer_gpu": True, "fallback_to_cpu": True}}, project_root=tmp_path)

    record = plan["records"][0]
    assert record["after"]["repair_result"] == "already_ok_cached_resolution"
    assert record["cached_runtime_resolution_check"]["status"] == "valid"
    assert record["cached_runtime_resolution_check"]["accelerator_cache_check"]["status"] == "valid"
    assert plan["summary"]["cached_runtime_resolutions"] == 1


def test_execute_repair_plan_reprobes_stale_onnx_provider_resolution(monkeypatch, tmp_path: Path):
    resolution_path = tmp_path / "Logs" / "dependency_resolution_onnx.json"
    resolution_path.parent.mkdir(parents=True)
    resolution_path.write_text(
        __import__("json").dumps(
            {
                "schema": "easy_asr_bench.runtime_resolution.v1",
                "dependency_group": "onnx",
                "status": "backend_usable",
                "backend_verified": True,
                "backend_probe_kind": "onnxruntime_provider_probe",
                "accelerator_requested": False,
                "versions": {},
                "providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
                "runtime_path": "",
                "config_runtime": {"provider": "auto", "prefer_gpu": True, "fallback_to_cpu": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.repair_plan.dependency_status",
        lambda config: {
            "onnx": {
                "available": True,
                "missing": [],
                "install_kind": "pip",
                "requirement_file": "requirements/onnx.txt",
                "recovery_command": "python -m pip install -r requirements/onnx.txt",
            },
        },
    )
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": False, "accelerator": None})
    monkeypatch.setattr("app.repair_plan.missing_modules_for_config", lambda group, config: [])
    monkeypatch.setattr("app.dependency_manager.onnxruntime_available_providers", lambda: (["CPUExecutionProvider"], ""))
    monkeypatch.setattr("app.repair_plan.backend_probe_for_group", lambda group, config: {"kind": "onnxruntime_provider_probe", "ok": True, "providers": ["CPUExecutionProvider"]})

    plan = execute_repair_plan({"runtime": {"provider": "auto", "prefer_gpu": True, "fallback_to_cpu": True}}, project_root=tmp_path)

    record = plan["records"][0]
    assert record["cached_runtime_resolution_check"]["status"] == "stale"
    assert "providers" in record["cached_runtime_resolution_check"]["mismatches"]
    assert record["cached_runtime_resolution_check"]["provider_mismatches"] == ["DmlExecutionProvider"]
    assert record["after"]["repair_result"] == "already_ok"
    assert plan["summary"]["cached_runtime_resolutions"] == 0


def test_execute_repair_plan_reuses_valid_llama_mtmd_runtime_path(monkeypatch, tmp_path: Path):
    cli = tmp_path / "llama-mtmd-cli.exe"
    cli.write_text("", encoding="utf-8")
    resolution_path = tmp_path / "Logs" / "dependency_resolution_llama_mtmd.json"
    resolution_path.parent.mkdir(parents=True)
    resolution_path.write_text(
        __import__("json").dumps(
            {
                "schema": "easy_asr_bench.runtime_resolution.v1",
                "dependency_group": "llama_mtmd",
                "status": "backend_usable",
                "backend_verified": True,
                "backend_probe_kind": "llama_mtmd_runtime_probe",
                "accelerator_requested": False,
                "versions": {},
                "providers": [],
                "runtime_path": str(cli),
                "config_runtime": {"provider": "auto", "prefer_gpu": True, "fallback_to_cpu": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.repair_plan.dependency_status",
        lambda config: {
            "llama_mtmd": {
                "available": True,
                "missing": [],
                "install_kind": "native_tool",
                "requirement_file": "",
                "recovery_command": "winget install -e --id ggml.llamacpp",
            },
        },
    )
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": False, "accelerator": None})
    monkeypatch.setattr("app.repair_plan.missing_modules_for_config", lambda group, config: [])
    monkeypatch.setattr("app.repair_plan.backend_probe_for_group", lambda group, config: (_ for _ in ()).throw(AssertionError("cached MTMD resolution should skip backend probe")))

    plan = execute_repair_plan({"runtime": {"provider": "auto", "prefer_gpu": True, "fallback_to_cpu": True}}, project_root=tmp_path)

    record = plan["records"][0]
    assert record["after"]["repair_result"] == "already_ok_cached_resolution"
    assert record["after"]["runtime_resolution"]["runtime_path"] == str(cli)
    assert record["cached_runtime_resolution_check"]["status"] == "valid"
    assert plan["summary"]["cached_runtime_resolutions"] == 1


def test_execute_repair_plan_refreshes_unstaged_llama_mtmd_runtime_path(monkeypatch, tmp_path: Path):
    cached_cli = tmp_path / "WinGetPackage" / "llama-mtmd-cli.exe"
    cached_cli.parent.mkdir()
    cached_cli.write_text("", encoding="utf-8")
    staged_cli = tmp_path / "Cache" / "native_tools" / "llama_mtmd" / "WinGetPackage" / "llama-mtmd-cli.exe"
    staged_cli.parent.mkdir(parents=True)
    staged_cli.write_text("", encoding="utf-8")
    resolution_path = tmp_path / "Logs" / "dependency_resolution_llama_mtmd.json"
    resolution_path.parent.mkdir(parents=True)
    resolution_path.write_text(
        __import__("json").dumps(
            {
                "schema": "easy_asr_bench.runtime_resolution.v1",
                "dependency_group": "llama_mtmd",
                "status": "backend_usable",
                "backend_verified": True,
                "backend_probe_kind": "llama_mtmd_runtime_probe",
                "accelerator_requested": False,
                "versions": {},
                "providers": [],
                "runtime_path": str(cached_cli),
                "config_runtime": {"provider": "auto", "prefer_gpu": True, "fallback_to_cpu": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.repair_plan.dependency_status",
        lambda config: {
            "llama_mtmd": {
                "available": True,
                "missing": [],
                "install_kind": "native_tool",
                "requirement_file": "",
                "recovery_command": "winget install -e --id ggml.llamacpp",
            },
        },
    )
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": False, "accelerator": None})
    monkeypatch.setattr("app.repair_plan.missing_modules_for_config", lambda group, config: [])
    monkeypatch.setattr(
        "app.repair_plan.backend_probe_for_group",
        lambda group, config: {
            "kind": "llama_mtmd_runtime_probe",
            "ok": True,
            "runtime_status": {"available": True, "path": str(staged_cli), "source": "staged_copy"},
        },
    )

    config = {
        "runtime": {"provider": "auto", "prefer_gpu": True, "fallback_to_cpu": True},
        "folders": {"cache": str(tmp_path / "Cache")},
    }
    plan = execute_repair_plan(config, project_root=tmp_path)

    record = plan["records"][0]
    assert record["cached_runtime_resolution_check"]["status"] == "stale"
    assert "runtime_path_not_staged" in record["cached_runtime_resolution_check"]["mismatches"]
    assert record["after"]["repair_result"] == "already_ok"
    assert record["after"]["runtime_resolution"]["runtime_path"] == str(staged_cli)
    assert plan["summary"]["cached_runtime_resolutions"] == 0


def test_execute_repair_plan_reuses_verified_llama_accelerator_resolution(monkeypatch, tmp_path: Path):
    resolution_path = tmp_path / "Logs" / "dependency_resolution_llama_cpp.json"
    resolution_path.parent.mkdir(parents=True)
    resolution_path.write_text(
        __import__("json").dumps(
            {
                "schema": "easy_asr_bench.runtime_resolution.v1",
                "dependency_group": "llama_cpp",
                "status": "accelerator_verified",
                "backend_verified": True,
                "backend_probe_kind": "llama_cpp_import_probe",
                "accelerator_requested": True,
                "accelerator": "vulkan",
                "accelerator_verified": True,
                "versions": {"llama_cpp": "0.3.18"},
                "providers": [],
                "runtime_path": "",
                "config_runtime": {"provider": "vulkan", "prefer_gpu": True, "fallback_to_cpu": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.repair_plan.dependency_status",
        lambda config: {
            "llama_cpp": {
                "available": True,
                "missing": [],
                "install_kind": "pip",
                "requirement_file": "requirements/llama_cpp_vulkan.txt",
                "recovery_command": "python -m pip install -r requirements/llama_cpp_vulkan.txt",
            },
        },
    )
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": True, "accelerator": "vulkan"})
    monkeypatch.setattr("app.repair_plan.missing_modules_for_config", lambda group, config: [])
    monkeypatch.setattr("app.repair_plan.importlib.metadata.version", lambda package: {"llama_cpp": "0.3.18"}[package])
    monkeypatch.setattr("app.dependency_manager.llama_cpp_gpu_capable", lambda: True)
    monkeypatch.setattr("app.repair_plan.backend_probe_for_group", lambda group, config: (_ for _ in ()).throw(AssertionError("verified llama.cpp accelerator resolution should skip backend probe")))

    plan = execute_repair_plan({"runtime": {"provider": "vulkan", "prefer_gpu": True, "fallback_to_cpu": True}}, project_root=tmp_path)

    record = plan["records"][0]
    assert record["after"]["repair_result"] == "already_ok_cached_resolution"
    assert record["cached_runtime_resolution_check"]["status"] == "valid"
    assert record["cached_runtime_resolution_check"]["accelerator_cache_check"]["status"] == "valid"
    assert plan["summary"]["cached_runtime_resolutions"] == 1


def test_execute_repair_plan_reprobes_unverified_llama_accelerator_resolution(monkeypatch, tmp_path: Path):
    resolution_path = tmp_path / "Logs" / "dependency_resolution_llama_cpp.json"
    resolution_path.parent.mkdir(parents=True)
    resolution_path.write_text(
        __import__("json").dumps(
            {
                "schema": "easy_asr_bench.runtime_resolution.v1",
                "dependency_group": "llama_cpp",
                "status": "backend_usable_accelerator_unverified",
                "backend_verified": True,
                "backend_probe_kind": "llama_cpp_import_probe",
                "accelerator_requested": True,
                "accelerator": "vulkan",
                "accelerator_verified": False,
                "versions": {"llama_cpp": "0.3.18"},
                "providers": [],
                "runtime_path": "",
                "config_runtime": {"provider": "vulkan", "prefer_gpu": True, "fallback_to_cpu": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.repair_plan.dependency_status",
        lambda config: {
            "llama_cpp": {
                "available": True,
                "missing": [],
                "install_kind": "pip",
                "requirement_file": "requirements/llama_cpp_vulkan.txt",
                "recovery_command": "python -m pip install -r requirements/llama_cpp_vulkan.txt",
            },
        },
    )
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": True, "accelerator": "vulkan"})
    monkeypatch.setattr("app.repair_plan.missing_modules_for_config", lambda group, config: [])
    monkeypatch.setattr("app.repair_plan.importlib.metadata.version", lambda package: {"llama_cpp": "0.3.18"}[package])
    monkeypatch.setattr(
        "app.repair_plan.backend_probe_for_group",
        lambda group, config: {
            "kind": "llama_cpp_import_probe",
            "ok": True,
            "accelerator_probe": {"requested": True, "accelerator": "vulkan", "ok": False},
        },
    )

    plan = execute_repair_plan({"runtime": {"provider": "vulkan", "prefer_gpu": True, "fallback_to_cpu": True}}, project_root=tmp_path)

    record = plan["records"][0]
    assert record["cached_runtime_resolution_check"]["status"] == "stale"
    assert "accelerator_unverified" in record["cached_runtime_resolution_check"]["mismatches"]
    assert record["cached_runtime_resolution_check"]["accelerator_cache_check"]["status"] == "stale"
    assert record["after"]["repair_result"] == "already_ok"
    assert plan["summary"]["cached_runtime_resolutions"] == 0


def test_validate_real_smoke_runs_repair_then_configured_rows(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"folders":{"models":"Models","input":"Input","output":"Output","temp":"Temp","logs":"Logs","cache":"Cache"},"runtime_validation":{"smoke_rows":["cpu_model_smoke","compare_html_offline"]}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("app.doctor.execute_repair_plan", lambda config, project_root=None, status=None: {"mode": "repair_all_safe", "summary": {"attempted": 0}})
    calls = []

    def fake_run(command, cwd=None, text=True, capture_output=True, timeout=None):
        calls.append(command)
        row_id = command[command.index("--row") + 1]
        row_dir = Path(command[command.index("--workdir") + 1]) / row_id
        row_dir.mkdir(parents=True, exist_ok=True)
        (row_dir / "row.json").write_text('{"status":"pass","summary":"ok"}\n', encoding="utf-8")
        return __import__("subprocess").CompletedProcess(command, 0, "stdout", "stderr")

    monkeypatch.setattr("app.doctor.subprocess.run", fake_run)

    report = run_real_smoke_validation(config_path, install_deps=True, allow_downloads=True)

    assert report["schema"] == "easy_asr_bench.real_smoke_validation.v1"
    assert report["repair_all_safe"]["mode"] == "repair_all_safe"
    assert report["summary"]["passed"] == 2
    assert [row["id"] for row in report["rows"]] == ["cpu_model_smoke", "compare_html_offline"]
    assert all("--install-deps" in command and "--allow-downloads" in command for command in calls)


def test_validate_real_smoke_no_network_suppresses_allow_downloads(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"folders":{"models":"Models","input":"Input","output":"Output","temp":"Temp","logs":"Logs","cache":"Cache"},"runtime_validation":{"smoke_rows":["real_media_download_cache"]}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("app.doctor.execute_repair_plan", lambda config, project_root=None, status=None: {"mode": "repair_all_safe", "summary": {"attempted": 0}})
    calls = []

    def fake_run(command, cwd=None, text=True, capture_output=True, timeout=None):
        calls.append(command)
        row_id = command[command.index("--row") + 1]
        row_dir = Path(command[command.index("--workdir") + 1]) / row_id
        row_dir.mkdir(parents=True, exist_ok=True)
        (row_dir / "row.json").write_text('{"status":"blocked","summary":"offline","block_reason":"network disabled","external_requirement":"rerun with --allow-downloads"}\n', encoding="utf-8")
        return __import__("subprocess").CompletedProcess(command, 0, "stdout", "stderr")

    monkeypatch.setattr("app.doctor.subprocess.run", fake_run)

    report = run_real_smoke_validation(config_path, allow_downloads=True, no_network=True)

    assert report["requested_allow_downloads"] is True
    assert report["allow_downloads"] is False
    assert report["no_network"] is True
    assert report["network_policy"] == "no_network"
    assert "--allow-downloads" not in calls[0]
    assert report["summary"]["blocked"] == 1


def test_validate_real_smoke_full_profile_runs_format_rows(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"folders":{"models":"Models","input":"Input","output":"Output","temp":"Temp","logs":"Logs","cache":"Cache"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("app.doctor.execute_repair_plan", lambda config, project_root=None, status=None: {"mode": "repair_all_safe", "summary": {"attempted": 0}})
    calls = []

    def fake_run(command, cwd=None, text=True, capture_output=True, timeout=None):
        calls.append(command)
        row_id = command[command.index("--row") + 1]
        row_dir = Path(command[command.index("--workdir") + 1]) / row_id
        row_dir.mkdir(parents=True, exist_ok=True)
        (row_dir / "row.json").write_text('{"status":"pass","summary":"ok"}\n', encoding="utf-8")
        return __import__("subprocess").CompletedProcess(command, 0, "stdout", "stderr")

    monkeypatch.setattr("app.doctor.subprocess.run", fake_run)

    report = run_real_smoke_validation(config_path, install_deps=True, allow_downloads=True, full_real_smoke=True)
    row_ids = [row["id"] for row in report["rows"]]

    assert report["smoke_profile"] == "full"
    assert report["full_real_smoke"] is True
    assert "real_public_media_faster_whisper_smollm_grading" in row_ids
    assert "real_public_media_openai_whisper_pt_smollm_grading" in row_ids
    assert "real_public_video_openai_whisper_pt_smollm_grading" in row_ids
    assert "real_public_media_whisper_cpp_ggml_smollm_grading" in row_ids
    assert "real_public_video_whisper_cpp_ggml_smollm_grading" in row_ids
    assert "real_public_media_hf_whisper_safetensors_smollm_grading_cpu" in row_ids
    assert "real_public_video_hf_whisper_safetensors_smollm_grading_cpu" in row_ids
    assert "real_public_media_generic_onnx_ctc_smollm_grading_cpu" in row_ids
    assert "real_public_video_generic_onnx_ctc_smollm_grading_cpu" in row_ids
    assert "real_public_media_gguf_asr_mmproj_smollm_grading" in row_ids
    assert "same_media_multi_model_smollm_benchmark_directml" in row_ids
    assert all("--install-deps" in command and "--allow-downloads" in command for command in calls)


def test_repair_backend_probe_checks_expected_onnx_provider(monkeypatch):
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": True, "accelerator": "directml"})
    monkeypatch.setattr("app.dependency_manager.onnxruntime_available_providers", lambda: (["CPUExecutionProvider"], ""))

    probe = backend_probe_for_group("onnx", {"runtime": {"provider": "directml"}})

    assert probe["kind"] == "onnxruntime_provider_probe"
    assert probe["ok"] is True
    assert probe["expected_provider"] == "DmlExecutionProvider"
    assert probe["expected_provider_present"] is False
    assert probe["accelerator_probe"]["requested"] is True
    assert probe["accelerator_probe"]["ok"] is False


def test_repair_backend_probe_records_ctranslate2_native_probe(monkeypatch):
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": False, "accelerator": None})
    captured = {}

    def fake_import_probe(modules, timeout=30):
        captured["timeout"] = timeout
        return {"ok": True, "loaded": [{"module": module} for module in modules]}

    monkeypatch.setattr("app.repair_plan._isolated_import_probe", fake_import_probe)
    monkeypatch.setattr("app.repair_plan.ctranslate2_probe", lambda: {"ok": True, "cuda_available": False, "version": "4.8.0"})

    probe = backend_probe_for_group("faster_whisper", {"runtime": {"provider": "cpu"}})

    assert probe["kind"] == "faster_whisper_ctranslate2_import_probe"
    assert probe["ok"] is True
    assert probe["ctranslate2"]["version"] == "4.8.0"
    assert [item["module"] for item in probe["imports"]["loaded"]] == ["faster_whisper", "ctranslate2", "pkg_resources"]
    assert captured["timeout"] == 90
    assert probe["accelerator_probe"]["requested"] is False
    assert probe["accelerator_probe"]["ok"] is True


def test_repair_backend_probe_records_media_tool_status(monkeypatch):
    monkeypatch.setattr(
        "app.dependency_manager.media_tools_status",
        lambda: {
            "available": True,
            "ffmpeg_path": "C:/tools/ffmpeg.exe",
            "ffprobe_available": False,
            "missing": [],
        },
    )

    probe = backend_probe_for_group("media_tools", {})

    assert probe["kind"] == "media_tools_ffmpeg_probe"
    assert probe["ok"] is True
    assert probe["runtime_status"]["ffmpeg_path"] == "C:/tools/ffmpeg.exe"


def test_repair_backend_probe_separates_llama_cpp_import_from_gpu_offload(monkeypatch):
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": True, "accelerator": "vulkan"})
    captured = {}

    def fake_import_probe(modules, timeout=30, pre_import_code=""):
        captured["pre_import_code"] = pre_import_code
        return {"ok": True, "loaded": [{"module": "llama_cpp", "version": "0.3.18"}]}

    monkeypatch.setattr("app.repair_plan._isolated_import_probe", fake_import_probe)
    monkeypatch.setattr("app.dependency_manager.llama_cpp_gpu_capable", lambda: False)

    probe = backend_probe_for_group("llama_cpp", {"runtime": {"provider": "auto", "prefer_gpu": True}})

    assert probe["kind"] == "llama_cpp_import_probe"
    assert probe["ok"] is True
    assert probe["gpu_offload_available"] is False
    assert probe["accelerator_probe"]["requested"] is True
    assert probe["accelerator_probe"]["accelerator"] == "vulkan"
    assert probe["accelerator_probe"]["ok"] is False
    assert "prepare_llama_cpp_dll_search_path" in captured["pre_import_code"]


def test_runtime_resolution_records_unverified_accelerator_fallback(tmp_path: Path):
    from app.repair_plan import build_runtime_resolution

    record = {
        "affected_dependency_group": "llama_cpp",
        "repair_command": "python -m pip install llama-cpp-python",
        "before": {"install_kind": "pip", "requirement_file": "requirements/llama_cpp.txt"},
        "after": {
            "repair_result": "already_ok",
            "backend_probe": {
                "kind": "llama_cpp_import_probe",
                "ok": True,
                "imports": {"loaded": [{"module": "llama_cpp", "version": "0.3.18"}]},
                "accelerator_probe": {
                    "requested": True,
                    "accelerator": "vulkan",
                    "ok": False,
                    "reason": "llama-cpp-python GPU/offload backend was not verified.",
                },
            },
        },
    }

    resolution = build_runtime_resolution(record, {"runtime": {"provider": "auto", "prefer_gpu": True, "fallback_to_cpu": True}}, tmp_path)

    assert resolution["schema"] == "easy_asr_bench.runtime_resolution.v1"
    assert resolution["status"] == "backend_usable_accelerator_unverified"
    assert resolution["backend_verified"] is True
    assert resolution["accelerator_requested"] is True
    assert resolution["accelerator_verified"] is False
    assert resolution["accelerator"] == "vulkan"
    assert resolution["versions"]["llama_cpp"] == "0.3.18"
    assert resolution["config_runtime"]["fallback_to_cpu"] is True


def test_runtime_resolution_check_accepts_matching_saved_resolution():
    from app.repair_plan import validate_saved_runtime_resolution

    saved = {
        "schema": "easy_asr_bench.runtime_resolution.v1",
        "dependency_group": "faster_whisper",
        "config_runtime": {"provider": "auto", "prefer_gpu": True, "fallback_to_cpu": True},
        "backend_probe_kind": "faster_whisper_ctranslate2_import_probe",
        "status": "backend_usable",
        "versions": {"faster_whisper": "1.2.1", "ctranslate2": "4.8.0"},
        "providers": [],
        "runtime_path": "",
        "accelerator": "",
        "accelerator_verified": False,
    }

    check = validate_saved_runtime_resolution(dict(saved), dict(saved))

    assert check["schema"] == "easy_asr_bench.runtime_resolution_check.v1"
    assert check["status"] == "valid"
    assert check["mismatches"] == []


def test_runtime_resolution_check_marks_version_or_config_change_stale():
    from app.repair_plan import validate_saved_runtime_resolution

    saved = {
        "schema": "easy_asr_bench.runtime_resolution.v1",
        "dependency_group": "faster_whisper",
        "config_runtime": {"provider": "auto", "prefer_gpu": True, "fallback_to_cpu": True},
        "backend_probe_kind": "faster_whisper_ctranslate2_import_probe",
        "status": "backend_usable",
        "versions": {"faster_whisper": "1.2.1", "ctranslate2": "4.7.2"},
        "providers": [],
        "runtime_path": "",
        "accelerator": "",
        "accelerator_verified": False,
    }
    current = dict(saved)
    current["config_runtime"] = {"provider": "cpu", "prefer_gpu": False, "fallback_to_cpu": True}
    current["versions"] = {"faster_whisper": "1.2.1", "ctranslate2": "4.8.0"}

    check = validate_saved_runtime_resolution(saved, current)

    assert check["status"] == "stale"
    assert set(check["mismatches"]) == {"config_runtime", "versions"}


def test_cuda_install_decision_requires_config_and_nvidia(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: True)

    disabled = cuda_install_decision(
        {"runtime": {"provider": "cuda"}, "dependency_install": {"allow_cuda_install": False}},
        "onnx",
    )
    enabled = cuda_install_decision(
        {"runtime": {"provider": "cuda"}, "dependency_install": {"allow_cuda_install": True}},
        "onnx",
    )

    assert disabled["use_cuda"] is False
    assert "allow_cuda_install" in disabled["reason"]
    assert enabled["use_cuda"] is True
    assert enabled["requirement_files"] == ["requirements/onnx_cuda.txt"]


def test_cuda_install_decision_requires_detected_nvidia(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: False)

    decision = cuda_install_decision(
        {"runtime": {"provider": "cuda"}, "dependency_install": {"allow_cuda_install": True}},
        "transformers_cpu",
    )

    assert decision["use_cuda"] is False
    assert "No NVIDIA GPU" in decision["reason"]


def test_onnx_uses_directml_when_windows_gpu_detected_without_nvidia(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.intel_gpu_or_npu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.windows_gpu_detected", lambda: True)

    decision = acceleration_install_decision(
        {"runtime": {"provider": "auto"}, "dependency_install": {"allow_accelerator_install": True, "allow_cuda_install": True}},
        "onnx",
    )

    assert decision["use_accelerator"] is True
    assert decision["accelerator"] == "directml"
    assert decision["requirement_files"] == ["requirements/onnx_directml.txt"]


def test_onnx_prefers_openvino_when_intel_detected(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.intel_gpu_or_npu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager.windows_gpu_detected", lambda: True)

    decision = acceleration_install_decision(
        {"runtime": {"provider": "auto"}, "dependency_install": {"allow_accelerator_install": True, "allow_cuda_install": True}},
        "onnx",
    )

    assert decision["use_accelerator"] is True
    assert decision["accelerator"] == "openvino"
    assert decision["requirement_files"] == ["requirements/onnx_openvino.txt"]


def test_onnx_uses_directml_for_amd_windows_gpu(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.intel_gpu_or_npu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.amd_gpu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager.windows_gpu_detected", lambda: True)

    decision = acceleration_install_decision(
        {"runtime": {"provider": "auto"}, "dependency_install": {"allow_accelerator_install": True, "allow_cuda_install": True}},
        "onnx",
    )

    assert decision["use_accelerator"] is True
    assert decision["accelerator"] == "directml"


def test_directml_replaces_plain_onnxruntime_requirement_and_flags_conflict(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.intel_gpu_or_npu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.windows_gpu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager._missing_import_modules", lambda metadata: [])
    monkeypatch.setattr("app.dependency_manager.distribution_installed", lambda package: package == "onnxruntime")
    monkeypatch.setattr("app.dependency_manager.onnx_provider_available", lambda provider: False)

    def fake_version(package):
        versions = {"onnxruntime": "1.26.0", "onnxruntime-directml": "1.24.4", "onnx": "1.21.0", "tokenizers": "0.22.1", "jinja2": "3.1.6"}
        return versions[package]

    monkeypatch.setattr("app.dependency_manager.importlib.metadata.version", fake_version)

    missing = missing_modules_for_config(
        "onnx",
        {"runtime": {"provider": "auto"}, "dependency_install": {"allow_accelerator_install": True}},
    )

    assert "onnxruntime<1.26,>=1.17 (installed 1.26.0)" not in missing
    assert "onnxruntime conflicts with directml package" in missing
    assert "onnxruntime DirectML provider" in missing


def test_auto_openvino_missing_provider_allows_cpu_fallback(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.intel_gpu_or_npu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager.windows_gpu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager._missing_import_modules", lambda metadata: [])
    monkeypatch.setattr("app.dependency_manager.requirement_version_issues", lambda requirement_files, ignored_packages=None: [])
    monkeypatch.setattr("app.dependency_manager.distribution_installed", lambda package: False)
    monkeypatch.setattr("app.dependency_manager.onnx_provider_available", lambda provider: False)

    missing_auto = missing_modules_for_config(
        "onnx",
        {"runtime": {"provider": "auto", "prefer_gpu": True, "fallback_to_cpu": True}, "dependency_install": {"allow_accelerator_install": True}},
    )
    missing_explicit = missing_modules_for_config(
        "onnx",
        {"runtime": {"provider": "openvino", "prefer_gpu": True, "fallback_to_cpu": True}, "dependency_install": {"allow_accelerator_install": True}},
    )

    assert "onnxruntime OpenVINO provider" not in missing_auto
    assert "onnxruntime OpenVINO provider" in missing_explicit


def test_llama_uses_vulkan_when_available_without_nvidia(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.vulkan_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager.vulkan_sdk_detected", lambda: True)

    decision = acceleration_install_decision(
        {"runtime": {"provider": "auto"}, "dependency_install": {"allow_accelerator_install": True, "allow_cuda_install": True}},
        "llama_cpp",
    )

    assert decision["use_accelerator"] is True
    assert decision["accelerator"] == "vulkan"
    assert decision["extra_index_url"].endswith("/vulkan")
    assert "llama-cpp-python" in decision["pip_args"]
    assert "--force-reinstall" in decision["pip_args"]
    assert "--index-url" in decision["pip_args"]


def test_llama_vulkan_prefers_prebuilt_wheel_without_sdk(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.vulkan_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager.vulkan_sdk_detected", lambda: False)

    decision = acceleration_install_decision(
        {"runtime": {"provider": "auto"}, "dependency_install": {"allow_accelerator_install": True, "allow_cuda_install": True}},
        "llama_cpp",
    )

    assert decision["use_accelerator"] is True
    assert decision["accelerator"] == "vulkan"
    assert "prebuilt wheel" in decision["reason"]


def test_auto_vulkan_does_not_make_cpu_llama_cpp_unavailable(monkeypatch):
    monkeypatch.setattr("app.dependency_manager._missing_import_modules", lambda metadata: [])
    monkeypatch.setattr("app.dependency_manager.requirement_version_issues", lambda requirement_files, ignored_packages=None: [])
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.vulkan_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager.llama_cpp_gpu_capable", lambda: False)

    missing_auto = missing_modules_for_config(
        "llama_cpp",
        {"runtime": {"provider": "auto", "prefer_gpu": True}, "dependency_install": {"allow_accelerator_install": True}},
    )
    missing_explicit = missing_modules_for_config(
        "llama_cpp",
        {"runtime": {"provider": "vulkan", "prefer_gpu": True}, "dependency_install": {"allow_accelerator_install": True}},
    )

    assert "llama-cpp-python GPU offload build" not in missing_auto
    assert "llama-cpp-python GPU offload build" in missing_explicit


def test_explicit_vulkan_provider_requires_sdk(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.vulkan_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.vulkan_sdk_detected", lambda: False)

    decision = acceleration_install_decision(
        {"runtime": {"provider": "vulkan"}, "dependency_install": {"allow_accelerator_install": True}},
        "llama_cpp",
    )

    assert decision["use_accelerator"] is False
    assert decision["accelerator"] == "vulkan"
    assert "Vulkan runtime" in decision["reason"]


def test_llama_cpp_cuda_resolver_maps_driver_to_wheel_tags(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.sys.version_info", (3, 12, 10))

    assert llama_cpp_cuda_tag_for_driver("530.30") == "cu121"
    assert llama_cpp_cuda_tag_for_driver("550.90") == "cu124"
    assert llama_cpp_cuda_tag_for_driver("555.99") == "cu125"
    assert llama_cpp_cuda_tag_for_driver("575.10") == "cu125"
    assert llama_cpp_cuda_tag_for_driver("580.00") == "cu125"


def test_llama_cpp_cuda_resolver_uses_selected_prebuilt_index(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.sys.version_info", (3, 12, 10))
    monkeypatch.setattr("app.dependency_manager.nvidia_driver_version", lambda: "575.10")

    decision = resolve_llama_cpp_wheel({"dependency_install": {}}, "cuda")

    assert decision.supported is True
    assert decision.extra_index_url.endswith("/cu125")
    assert decision.pip_args == ("--upgrade", "--force-reinstall", "--no-deps", "--index-url", decision.extra_index_url, "llama-cpp-python")


def test_llama_cpp_cuda_python_313_falls_back_cpu(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.sys.version_info", (3, 13, 0))

    decision = resolve_llama_cpp_wheel({"dependency_install": {}}, "cuda")

    assert decision.supported is False
    assert "Python 3.13" in decision.reason


def test_repair_command_uses_windows_env_syntax_for_vulkan_source_build(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.sys.version_info", (3, 12, 10))
    monkeypatch.setattr("app.dependency_manager.vulkan_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.vulkan_build_tooling_detected", lambda: True)

    command = recovery_command_for_config(
        "llama_cpp",
        {
            "runtime": {"provider": "vulkan"},
            "dependency_install": {"allow_accelerator_install": True, "allow_vulkan_source_build": True},
        },
    )

    assert "PowerShell:" in command
    assert "$env:CMAKE_ARGS" in command
    assert "cmd.exe:" in command
    assert 'set "CMAKE_ARGS=' in command
    assert "CMAKE_ARGS=-DGGML_VULKAN=on " not in command


def test_doctor_prints_dependency_repair_and_cuda_status(monkeypatch, tmp_path: Path, capsys):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(
        "app.doctor.load_config",
        lambda path: {
            "folders": {
                "models": str(tmp_path / "Models"),
                "input": str(tmp_path / "Input"),
                "output": str(tmp_path / "Output"),
                "temp": str(tmp_path / "Temp"),
                "logs": str(tmp_path / "Logs"),
                "cache": str(tmp_path / "Cache"),
            }
        },
    )
    monkeypatch.setattr(
        "app.doctor.dependency_status",
        lambda config: {
            "core": {
                "available": True,
                "missing": [],
                "description": "base app",
                "recovery_command": '"python" -m pip install -r requirements/core.txt',
            },
            "onnx": {
                "available": False,
                "missing": ["onnxruntime"],
                "description": "ONNX ASR",
                "recovery_command": '"python" -m pip install -r requirements/onnx.txt',
                "cuda_recovery_command": '"python" -m pip install -r requirements/onnx_cuda.txt',
            },
        },
    )
    monkeypatch.setattr(
        "app.doctor.cuda_diagnostics",
        lambda: {
            "torch_installed": True,
            "torch_cuda_available": False,
            "torch_cuda_version": None,
            "torch_gpu_names": [],
            "onnxruntime_installed": True,
            "onnxruntime_providers": ["CPUExecutionProvider"],
            "nvidia_gpu_detected": True,
            "amd_gpu_detected": False,
            "intel_gpu_or_npu_detected": False,
            "windows_gpu_detected": True,
            "vulkan_detected": False,
            "vulkan_sdk_detected": False,
            "huggingface_cache": {
                "cache_dir": str(tmp_path / "hf" / "hub"),
                "symlink_supported": False,
                "repair_guidance": "Enable Windows Developer Mode or use a roomy cache disk.",
            },
            "messages": ["ONNX Runtime is installed, but CUDAExecutionProvider is not available. ONNX models will run on CPUExecutionProvider."],
        },
    )

    assert run_doctor(config_path, strict=True) == 0

    output = capsys.readouterr().out
    assert "MISSING  onnx - ONNX ASR" in output
    assert "repair:" in output
    assert "CUDA/provider checks:" in output
    assert "torch CUDA available: False" in output
    assert "Vulkan SDK detected: False" in output
    assert "Hugging Face cache dir:" in output
    assert "Hugging Face cache symlink support: False" in output
    assert "CPUExecutionProvider" in output


def test_doctor_repair_plan_json_output(monkeypatch, tmp_path: Path, capsys):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(
        "app.doctor.load_config",
        lambda path: {
            "folders": {
                "models": str(tmp_path / "Models"),
                "input": str(tmp_path / "Input"),
                "output": str(tmp_path / "Output"),
                "temp": str(tmp_path / "Temp"),
                "logs": str(tmp_path / "Logs"),
                "cache": str(tmp_path / "Cache"),
            }
        },
    )
    monkeypatch.setattr(
        "app.doctor.dependency_status",
        lambda config: {
            "core": {"available": True, "missing": [], "description": "core", "recovery_command": "repair core"},
            "onnx": {"available": False, "missing": ["onnxruntime"], "description": "onnx", "recovery_command": "repair onnx"},
        },
    )
    monkeypatch.setattr(
        "app.doctor.cuda_diagnostics",
        lambda: {
            "torch_installed": False,
            "torch_cuda_available": False,
            "torch_cuda_version": None,
            "torch_gpu_names": [],
            "onnxruntime_installed": False,
            "onnxruntime_providers": [],
            "nvidia_gpu_detected": False,
            "messages": [],
        },
    )
    monkeypatch.setattr("app.repair_plan.acceleration_install_decision", lambda config, group: {"use_accelerator": False, "accelerator": None})

    assert run_doctor(config_path, repair_plan_output=True) == 0

    output = capsys.readouterr().out
    plan = __import__("json").loads(output)
    assert plan["schema"] == "easy_asr_bench.repair_plan.v1"
    assert plan["summary"]["needs_repair"] == 1
    assert any(record["affected_dependency_group"] == "onnx" for record in plan["records"])


def test_doctor_repair_all_safe_json_output(monkeypatch, tmp_path: Path, capsys):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(
        "app.doctor.load_config",
        lambda path: {
            "folders": {
                "models": str(tmp_path / "Models"),
                "input": str(tmp_path / "Input"),
                "output": str(tmp_path / "Output"),
                "temp": str(tmp_path / "Temp"),
                "logs": str(tmp_path / "Logs"),
                "cache": str(tmp_path / "Cache"),
            }
        },
    )
    monkeypatch.setattr("app.doctor.dependency_status", lambda config: {"core": {"available": True, "missing": [], "description": "core", "recovery_command": "repair core"}})
    monkeypatch.setattr(
        "app.doctor.cuda_diagnostics",
        lambda: {
            "torch_installed": False,
            "torch_cuda_available": False,
            "torch_cuda_version": None,
            "torch_gpu_names": [],
            "onnxruntime_installed": False,
            "onnxruntime_providers": [],
            "nvidia_gpu_detected": False,
            "messages": [],
        },
    )
    monkeypatch.setattr(
        "app.doctor.execute_repair_plan",
        lambda config, status=None: {
            "schema": "easy_asr_bench.repair_plan.v1",
            "mode": "repair_all_safe",
            "records": [],
            "summary": {"attempted": 0},
        },
    )

    assert run_doctor(config_path, repair_all_safe=True) == 0

    output = capsys.readouterr().out
    plan = __import__("json").loads(output)
    assert plan["schema"] == "easy_asr_bench.repair_plan.v1"
    assert plan["mode"] == "repair_all_safe"
    assert plan["summary"]["attempted"] == 0


def test_cuda_requested_warning_reports_cpu_fallback(monkeypatch, capsys):
    monkeypatch.setattr(
        "app.dependency_manager.cuda_diagnostics",
        lambda: {
            "messages": [
                "Torch is installed, but Torch CUDA is not available. Transformers models will run on CPU.",
                "ONNX Runtime is installed, but CUDAExecutionProvider is not available. ONNX models will run on CPUExecutionProvider.",
            ]
        },
    )

    warn_runtime_dependency_fallbacks({"runtime": {"provider": "cuda", "prefer_gpu": False}})

    output = capsys.readouterr().out
    assert "CUDA was requested or preferred" in output
    assert "Transformers models will run on CPU" in output
    assert "ONNX models will run on CPUExecutionProvider" in output


def test_cuda_warning_silent_when_gpu_not_requested(capsys):
    warn_runtime_dependency_fallbacks({"runtime": {"provider": "auto", "prefer_gpu": False}})

    assert capsys.readouterr().out == ""


def test_dependency_install_prompt_has_explicit_recovery_choices(monkeypatch, capsys):
    answers = iter(["r", "q"])

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("app.main.choose_one", lambda *args, **kwargs: None)
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    decision = _dependency_install_confirmation("onnx", "python -m pip install -r requirements/onnx.txt")

    output = capsys.readouterr().out
    assert decision == "quit"
    assert "Manual repair command: python -m pip install -r requirements/onnx.txt" in output


def test_dependency_install_prompt_skip_means_skip_affected_models(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("app.main.choose_one", lambda *args, **kwargs: None)
    monkeypatch.setattr("builtins.input", lambda prompt: "s")

    assert _dependency_install_confirmation("onnx", "repair") == "skip_group"


def test_optional_install_failure_prints_repair_and_skips_only_affected(monkeypatch, tmp_path: Path, capsys):
    from app.adapters.base import ModelCandidate

    class FakeAdapter:
        name = "fake"

        def required_dependency_groups(self, candidate):
            return list(candidate.metadata.get("groups", []))

    good = ModelCandidate(
        candidate_id="good",
        display_name="Good model",
        family_name="Good",
        backend="test",
        container_format="test",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="fp32",
        path=tmp_path / "good",
        adapter_name="fake",
        runnable=True,
        metadata={"groups": []},
    )
    bad = ModelCandidate(
        candidate_id="bad",
        display_name="Bad model",
        family_name="Bad",
        backend="test",
        container_format="test",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="fp32",
        path=tmp_path / "bad",
        adapter_name="fake",
        runnable=True,
        metadata={"groups": ["onnx"]},
    )

    monkeypatch.setattr("app.main.adapter_for", lambda candidate: FakeAdapter())
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "s")
    monkeypatch.setattr("app.dependency_manager.missing_modules_for_config", lambda group, config: ["onnxruntime"] if group == "onnx" else [])
    monkeypatch.setattr("app.dependency_manager.install_group_for_config", lambda group, root, config, log_path=None: (_ for _ in ()).throw(RuntimeError("pip failed")))

    kept, _ = ensure_dependencies([good, bad], {"dependency_install": {"auto_install_missing_runtime_dependencies": True}})

    output = capsys.readouterr().out
    assert kept == [good]
    assert "Skipped dependency install for onnx" in output
    assert "Skipping Bad model" in output
    assert "Good model" not in output


def test_disabled_auto_install_skips_only_models_with_missing_groups(monkeypatch, tmp_path: Path, capsys):
    from app.adapters.base import ModelCandidate

    class FakeAdapter:
        name = "fake"

        def required_dependency_groups(self, candidate):
            return list(candidate.metadata.get("groups", []))

    good = ModelCandidate(
        candidate_id="good",
        display_name="Good model",
        family_name="Good",
        backend="test",
        container_format="test",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="fp32",
        path=tmp_path / "good",
        adapter_name="fake",
        runnable=True,
        metadata={"groups": []},
    )
    bad = ModelCandidate(
        candidate_id="bad",
        display_name="Bad model",
        family_name="Bad",
        backend="test",
        container_format="test",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="fp32",
        path=tmp_path / "bad",
        adapter_name="fake",
        runnable=True,
        metadata={"groups": ["onnx"]},
    )

    monkeypatch.setattr("app.main.adapter_for", lambda candidate: FakeAdapter())
    monkeypatch.setattr("app.dependency_manager.missing_modules_for_config", lambda group, config: ["onnxruntime"] if group == "onnx" else [])

    kept, _ = ensure_dependencies(
        [good, bad],
        {"dependency_install": {"auto_install_missing_runtime_dependencies": False}},
    )

    output = capsys.readouterr().out
    assert kept == [good]
    assert "Automatic dependency repair is disabled" in output
    assert "Skipping Bad model" in output


def test_reference_llm_dependency_failure_does_not_drop_asr_models(monkeypatch, tmp_path: Path, capsys):
    from app.adapters.base import ModelCandidate

    class FakeAdapter:
        name = "fake"

        def required_dependency_groups(self, candidate):
            return list(candidate.metadata.get("groups", []))

    asr = ModelCandidate(
        candidate_id="asr",
        display_name="ASR model",
        family_name="ASR",
        backend="test",
        container_format="test",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="fp32",
        path=tmp_path / "asr",
        adapter_name="fake",
        runnable=True,
        metadata={"groups": []},
    )
    llm = ModelCandidate(
        candidate_id="llm",
        display_name="Reference LLM",
        family_name="LLM",
        backend="llama.cpp",
        container_format="gguf",
        task="text-generation",
        precision="q4",
        quantization_label="Q4",
        path=tmp_path / "llm.gguf",
        adapter_name="fake",
        runnable=True,
        category="reference_llm",
        metadata={"groups": ["llama_cpp"]},
    )

    monkeypatch.setattr("app.main.adapter_for", lambda candidate: FakeAdapter())
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "s")
    monkeypatch.setattr("app.dependency_manager.missing_modules_for_config", lambda group, config: ["llama_cpp"] if group == "llama_cpp" else [])
    monkeypatch.setattr("app.dependency_manager.install_group_for_config", lambda group, root, config, log_path=None: (_ for _ in ()).throw(RuntimeError("pip failed")))

    kept, kept_llm = ensure_dependencies(
        [asr],
        {"dependency_install": {"auto_install_missing_runtime_dependencies": True}},
        reference_llm=llm,
    )

    output = capsys.readouterr().out
    assert kept == [asr]
    assert kept_llm is None
    assert "Skipping Reference LLM" in output
    assert "ASR model" not in output


def test_optional_install_uses_cuda_requirements_when_allowed(monkeypatch, tmp_path: Path, capsys):
    from app.adapters.base import ModelCandidate

    class FakeAdapter:
        name = "fake"

        def required_dependency_groups(self, candidate):
            return ["onnx"]

    model = ModelCandidate(
        candidate_id="model",
        display_name="CUDA model",
        family_name="CUDA",
        backend="test",
        container_format="test",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="fp32",
        path=tmp_path / "model",
        adapter_name="fake",
        runnable=True,
    )
    calls = []
    missing = {"before": True}

    monkeypatch.setattr("app.main.adapter_for", lambda candidate: FakeAdapter())
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager.onnx_provider_available", lambda provider: not missing["before"])

    def fake_missing(group):
        if group == "onnx" and missing["before"]:
            return ["onnxruntime"]
        return []

    def fake_install(group, root, config, log_path=None):
        decision = acceleration_install_decision(config, group)
        calls.append((group, decision["use_accelerator"], decision["accelerator"]))
        missing["before"] = False

    monkeypatch.setattr("app.dependency_manager.missing_modules_for_config", lambda group, config: fake_missing(group))
    monkeypatch.setattr("app.dependency_manager.install_group_for_config", fake_install)

    kept, _ = ensure_dependencies(
        [model],
        {
            "runtime": {"provider": "cuda", "prefer_gpu": True},
            "dependency_install": {"auto_install_missing_runtime_dependencies": True, "allow_cuda_install": True},
        },
    )

    output = capsys.readouterr().out
    assert kept == [model]
    assert calls == [("onnx", True, "cuda")]
    assert "CUDA install:" in output


def test_optional_install_empty_input_installs_and_failures_skip_only_affected(monkeypatch, tmp_path: Path, capsys):
    from app.adapters.base import ModelCandidate

    class FakeAdapter:
        name = "fake"

        def required_dependency_groups(self, candidate):
            return list(candidate.metadata.get("groups", []))

    good = ModelCandidate(
        candidate_id="good",
        display_name="Good model",
        family_name="Good",
        backend="test",
        container_format="test",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="fp32",
        path=tmp_path / "good",
        adapter_name="fake",
        runnable=True,
        metadata={"groups": []},
    )
    bad = ModelCandidate(
        candidate_id="bad",
        display_name="Bad model",
        family_name="Bad",
        backend="test",
        container_format="test",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="fp32",
        path=tmp_path / "bad",
        adapter_name="fake",
        runnable=True,
        metadata={"groups": ["onnx"]},
    )

    monkeypatch.setattr("app.main.adapter_for", lambda candidate: FakeAdapter())
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    monkeypatch.setattr("app.dependency_manager.missing_modules_for_config", lambda group, config: ["onnxruntime"] if group == "onnx" else [])
    monkeypatch.setattr("app.dependency_manager.install_group_for_config", lambda group, root, config, log_path=None: (_ for _ in ()).throw(RuntimeError("pip failed")))

    kept, _ = ensure_dependencies([good, bad], {"dependency_install": {"auto_install_missing_runtime_dependencies": True}})

    output = capsys.readouterr().out
    assert kept == [good]
    assert "Install failed for onnx" in output
    assert "Manual repair command:" in output
    assert "Skipping Bad model" in output


def test_optional_install_noninteractive_skips_without_prompting(monkeypatch, tmp_path: Path, capsys):
    from app.adapters.base import ModelCandidate

    class FakeAdapter:
        name = "fake"

        def required_dependency_groups(self, candidate):
            return list(candidate.metadata.get("groups", []))

    bad = ModelCandidate(
        candidate_id="bad",
        display_name="Bad model",
        family_name="Bad",
        backend="test",
        container_format="test",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="fp32",
        path=tmp_path / "bad",
        adapter_name="fake",
        runnable=True,
        metadata={"groups": ["onnx"]},
    )
    prompted = {"value": False}

    monkeypatch.setattr("app.main.adapter_for", lambda candidate: FakeAdapter())
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt: prompted.update(value=True) or "")
    monkeypatch.setattr("app.dependency_manager.missing_modules_for_config", lambda group, config: ["onnxruntime"] if group == "onnx" else [])

    kept, _ = ensure_dependencies([bad], {"dependency_install": {"auto_install_missing_runtime_dependencies": True}})

    output = capsys.readouterr().out
    assert kept == []
    assert prompted["value"] is False
    assert "noninteractive input cannot confirm" in output


def test_cuda_repair_triggers_for_cpu_only_torch(monkeypatch):
    from app.dependency_manager import missing_modules_for_config

    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager._missing_import_modules", lambda metadata: [])
    monkeypatch.setattr("app.dependency_manager.torch_cuda_available", lambda: False)

    missing = missing_modules_for_config(
        "transformers_cpu",
        {
            "runtime": {"provider": "auto", "prefer_gpu": True},
            "dependency_install": {"allow_cuda_install": True},
        },
    )

    assert missing == ["torch CUDA wheel"]


def test_faster_whisper_cuda_repair_requires_ctranslate2_backend(monkeypatch):
    from app.dependency_manager import missing_modules_for_config

    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager._missing_import_modules", lambda metadata: [])
    monkeypatch.setattr(
        "app.dependency_manager.module_available",
        lambda module: module not in {"nvidia.cublas.lib", "nvidia.cublas.bin", "nvidia.cudnn.lib", "nvidia.cudnn.bin"},
    )
    monkeypatch.setattr("app.dependency_manager.ctranslate2_cuda_available", lambda: False)

    missing = missing_modules_for_config(
        "faster_whisper",
        {
            "runtime": {"provider": "auto", "prefer_gpu": True},
            "dependency_install": {"allow_cuda_install": True},
        },
    )

    assert "CTranslate2 CUDA backend" in missing


def test_faster_whisper_cuda_accepts_modern_nvidia_bin_packages(monkeypatch):
    from app.dependency_manager import missing_modules_for_config

    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager._missing_import_modules", lambda metadata: [])
    monkeypatch.setattr(
        "app.dependency_manager.module_available",
        lambda module: module in {"nvidia.cublas.bin", "nvidia.cudnn.bin"},
    )
    monkeypatch.setattr("app.dependency_manager.ctranslate2_cuda_available", lambda: True)

    missing = missing_modules_for_config(
        "faster_whisper",
        {
            "runtime": {"provider": "auto", "prefer_gpu": True},
            "dependency_install": {"allow_cuda_install": True},
        },
    )

    assert missing == []


def test_cuda_repair_triggers_for_llama_cpp_without_gpu_offload(monkeypatch):
    from app.dependency_manager import missing_modules_for_config

    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager._missing_import_modules", lambda metadata: [])
    monkeypatch.setattr("app.dependency_manager.llama_cpp_gpu_capable", lambda: False)

    missing = missing_modules_for_config(
        "llama_cpp",
        {
            "runtime": {"provider": "auto", "prefer_gpu": True},
            "dependency_install": {"allow_cuda_install": True},
        },
    )

    assert missing == ["llama-cpp-python GPU offload build"]


def test_llama_cuda_install_falls_back_to_cpu_when_wheel_index_unavailable(tmp_path, monkeypatch):
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements" / "llama_cpp.txt").write_text("llama-cpp-python\n", encoding="utf-8")
    (tmp_path / "requirements" / "llama_cpp_cuda_cu124.txt").write_text(
        "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124\nllama-cpp-python\n",
        encoding="utf-8",
    )
    calls = []

    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager.llama_cpp_wheel_index_available", lambda url: False)
    monkeypatch.setattr("app.dependency_manager.subprocess.check_call", lambda command, env=None: calls.append(command))

    decision = install_group_for_config(
        "llama_cpp",
        tmp_path,
        {
            "runtime": {"provider": "cuda", "prefer_gpu": True},
            "dependency_install": {"allow_cuda_install": True, "allow_accelerator_install": True},
        },
    )

    assert decision["use_accelerator"] is False
    assert decision["accelerator_fallback"] == "cpu"
    assert "unavailable" in decision["accelerator_fallback_reason"]
    assert decision["cpu_wheel_fallback"] is True
    assert calls == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--extra-index-url",
            "https://abetlen.github.io/llama-cpp-python/whl/cpu",
            "llama-cpp-python",
        ]
    ]


def test_llama_cpp_cpu_fallback_uses_prebuilt_wheel_index(tmp_path, monkeypatch):
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements" / "llama_cpp.txt").write_text("llama-cpp-python\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr("app.dependency_manager.vulkan_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.llama_cpp_wheel_index_available", lambda url: False)
    monkeypatch.setattr("app.dependency_manager.subprocess.check_call", lambda command, env=None: calls.append(command))

    decision = install_group_for_config(
        "llama_cpp",
        tmp_path,
        {"runtime": {"provider": "cpu", "prefer_gpu": False}, "dependency_install": {"allow_accelerator_install": True}},
    )

    assert decision["cpu_wheel_fallback"] is True
    assert decision["cpu_wheel_index"].endswith("/whl/cpu")
    assert calls == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--extra-index-url",
            "https://abetlen.github.io/llama-cpp-python/whl/cpu",
            "llama-cpp-python",
        ]
    ]


def test_directml_install_removes_conflicting_plain_onnxruntime(tmp_path, monkeypatch):
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements" / "onnx_directml.txt").write_text("onnxruntime-directml\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.intel_gpu_or_npu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.windows_gpu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager.distribution_installed", lambda package: package == "onnxruntime")
    monkeypatch.setattr("app.dependency_manager.subprocess.check_call", lambda command, env=None: calls.append(command))
    provider_results = iter([([], "broken"), (["DmlExecutionProvider"], "")])
    monkeypatch.setattr("app.dependency_manager.onnxruntime_available_providers", lambda: next(provider_results))

    decision = install_group_for_config(
        "onnx",
        tmp_path,
        {"runtime": {"provider": "auto"}, "dependency_install": {"allow_accelerator_install": True}},
    )

    assert decision["accelerator"] == "directml"
    assert decision["provider_compatibility_repair"] == "onnxruntime-directml==1.24.4"
    assert calls[0][-3:] == ["uninstall", "-y", "onnxruntime"]
    assert calls[1][-2:] == ["-r", str(tmp_path / "requirements" / "onnx_directml.txt")]
    assert calls[2][-4:] == ["--upgrade", "--force-reinstall", "--no-deps", "onnxruntime-directml==1.24.4"]
    assert calls[3][-1] == "onnxruntime-directml==1.24.4"


def test_openvino_provider_repair_skips_unavailable_versions(tmp_path, monkeypatch):
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements" / "onnx_openvino.txt").write_text("onnxruntime-openvino\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.intel_gpu_or_npu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager.distribution_installed", lambda package: False)
    provider_results = iter([([], "missing"), (["OpenVINOExecutionProvider"], "")])
    monkeypatch.setattr("app.dependency_manager.onnxruntime_available_providers", lambda: next(provider_results))

    def fake_check_call(command, env=None):
        calls.append(command)
        if command[-1] == "onnxruntime-openvino==1.24.1":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("app.dependency_manager.subprocess.check_call", fake_check_call)

    decision = install_group_for_config(
        "onnx",
        tmp_path,
        {
            "runtime": {"provider": "openvino"},
            "dependency_install": {
                "allow_accelerator_install": True,
                "onnxruntime_openvino_compatibility_versions": ["1.24.1", "1.23.0"],
            },
        },
    )

    assert decision["provider_compatibility_repair"] == "onnxruntime-openvino==1.23.0"
    assert any(command[-1] == "onnxruntime-openvino==1.24.1" for command in calls)
    assert any(command[-1] == "onnxruntime-openvino==1.23.0" for command in calls)


def test_runtime_environment_persists_cuda_diagnostics(monkeypatch):
    monkeypatch.setattr(
        "app.results_writer.cuda_diagnostics",
        lambda: {
            "torch_installed": True,
            "torch_cuda_available": False,
            "onnxruntime_providers": ["CPUExecutionProvider"],
            "messages": ["CPU fallback"],
        },
    )

    environment = runtime_environment()

    assert environment["cuda_diagnostics"]["torch_cuda_available"] is False
    assert environment["cuda_diagnostics"]["messages"] == ["CPU fallback"]


def test_dependency_resolution_environment_summarizes_saved_runtime_evidence(tmp_path: Path):
    logs = tmp_path / "Logs"
    logs.mkdir()
    (logs / "dependency_resolution_llama_cpp.json").write_text(
        __import__("json").dumps(
            {
                "schema": "easy_asr_bench.runtime_resolution.v1",
                "dependency_group": "llama_cpp",
                "status": "backend_usable_accelerator_unverified",
                "backend_verified": True,
                "backend_probe_kind": "llama_cpp_import_probe",
                "accelerator_requested": True,
                "accelerator": "vulkan",
                "accelerator_verified": False,
                "versions": {"llama-cpp-python": "0.3.16"},
                "providers": [],
                "runtime_path": "",
                "config_runtime": {"provider": "auto", "prefer_gpu": True, "fallback_to_cpu": True},
            }
        ),
        encoding="utf-8",
    )
    (logs / "repair_all_safe_last.json").write_text(
        __import__("json").dumps(
            {
                "mode": "repair_all_safe",
                "summary": {
                    "runtime_resolutions": 1,
                    "cached_runtime_resolutions": 1,
                    "previous_runtime_resolution_stale": 0,
                },
                "records": [
                    {
                        "affected_dependency_group": "llama_cpp",
                        "status": "ok",
                        "repair_action": "verify_cached",
                        "cached_runtime_resolution_check": {"status": "valid"},
                        "after": {
                            "repair_result": "already_ok_cached_resolution",
                            "runtime_resolution_path": str(logs / "dependency_resolution_llama_cpp.json"),
                            "previous_runtime_resolution_check": {"status": "valid"},
                            "backend_probe": {
                                "kind": "cached_runtime_resolution",
                                "accelerator_probe": {"requested": True, "accelerator": "vulkan", "ok": False},
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    environment = dependency_resolution_environment(tmp_path)

    assert environment["summary"]["resolution_count"] == 1
    assert environment["summary"]["accelerator_unverified"] == 1
    assert environment["resolutions"][0]["dependency_group"] == "llama_cpp"
    assert environment["last_repair_all_safe"]["summary"]["cached_runtime_resolutions"] == 1
    assert environment["last_repair_all_safe"]["records"][0]["cached_runtime_resolution_status"] == "valid"
    assert environment["last_repair_all_safe"]["records"][0]["repair_action"] == "verify_cached"
    assert environment["last_repair_all_safe"]["records"][0]["repair_result"] == "already_ok_cached_resolution"


def test_faster_whisper_native_repair_uses_requirement_probe_before_candidate_fallback(monkeypatch, tmp_path: Path):
    from app.adapters.base import ModelCandidate
    from app.main import _repair_faster_whisper_native_stack

    model = ModelCandidate(
        candidate_id="fw",
        display_name="FW",
        family_name="FW",
        backend="faster-whisper",
        container_format="ctranslate2",
        task="automatic-speech-recognition",
        precision="unknown",
        quantization_label="unknown",
        path=tmp_path / "model",
        adapter_name="faster_whisper",
        runnable=True,
    )
    attempts: list[list[str]] = []

    def fake_run(command, cwd=None, text=True, stdout=None, stderr=None):
        attempts.append(command)
        return __import__("subprocess").CompletedProcess(command, 0, "installed")

    probe_results = iter([""])

    monkeypatch.setattr("app.main.subprocess.run", fake_run)
    monkeypatch.setattr("app.adapters.faster_whisper_asr.probe_faster_whisper_load", lambda path, device, compute_type: next(probe_results))

    error = _repair_faster_whisper_native_stack(
        model,
        {
            "folders": {"logs": str(tmp_path / "Logs")},
            "runtime": {"provider": "cpu", "prefer_gpu": False},
            "dependency_install": {},
        },
        "initial crash",
    )

    assert error == ""
    assert attempts == [[__import__("sys").executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "-r", str(Path(__file__).resolve().parents[1] / "requirements" / "faster_whisper.txt")]]


def test_faster_whisper_native_repair_discovers_candidates_after_requirement_probe_fails(monkeypatch, tmp_path: Path):
    from app.adapters.base import ModelCandidate
    from app.main import _repair_faster_whisper_native_stack

    model = ModelCandidate(
        candidate_id="fw",
        display_name="FW",
        family_name="FW",
        backend="faster-whisper",
        container_format="ctranslate2",
        task="automatic-speech-recognition",
        precision="unknown",
        quantization_label="unknown",
        path=tmp_path / "model",
        adapter_name="faster_whisper",
        runnable=True,
    )
    attempts: list[list[str]] = []

    def fake_run(command, cwd=None, text=True, stdout=None, stderr=None, timeout=None):
        attempts.append(command)
        if command[:4] == [__import__("sys").executable, "-m", "pip", "index"]:
            return __import__("subprocess").CompletedProcess(command, 0, "Available versions: 4.10.0, 4.9.0, 4.8.0, 4.7.2, 4.3.1\n")
        return __import__("subprocess").CompletedProcess(command, 0, "installed")

    probe_results = iter(["still broken", "candidate broken", ""])

    monkeypatch.setattr("app.main.subprocess.run", fake_run)
    monkeypatch.setattr("app.adapters.faster_whisper_asr.probe_faster_whisper_load", lambda path, device, compute_type: next(probe_results))

    error = _repair_faster_whisper_native_stack(
        model,
        {
            "folders": {"logs": str(tmp_path / "Logs")},
            "runtime": {"provider": "cpu", "prefer_gpu": False},
            "dependency_install": {},
        },
        "initial crash",
    )

    assert error == ""
    install_specs = [part for command in attempts for part in command if part.startswith("ctranslate2==")]
    assert install_specs == ["ctranslate2==4.8.0", "ctranslate2==4.7.2"]
    assert all("4.9.0" not in spec and "4.10.0" not in spec and "4.3.1" not in spec for spec in install_specs)


def test_faster_whisper_configured_ctranslate2_candidates_are_requirement_bounded(monkeypatch, tmp_path: Path):
    from app.main import _discover_ctranslate2_candidate_versions

    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements" / "faster_whisper.txt").write_text(
        "faster-whisper>=1.2,<1.3\nctranslate2>=4.4,<4.9\nsetuptools>=65,<81\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("app.main._installed_distribution_version", lambda package: "")

    class Log:
        def __init__(self) -> None:
            self.text = ""

        def write(self, text: str) -> None:
            self.text += text

    log = Log()
    versions = _discover_ctranslate2_candidate_versions(
        {"dependency_install": {"ctranslate2_compatibility_versions": ["4.10.0", "4.8.1", "4.4.0", "4.3.9"]}},
        tmp_path,
        log,
    )

    assert versions == ["4.8.1", "4.4.0"]
    assert "pip index" not in log.text
