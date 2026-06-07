from pathlib import Path

from app.config import DEFAULT_CONFIG
from app.dependency_manager import ACCELERATOR_OVERRIDES, CUDA_INSTALL_OVERRIDES, acceleration_install_decision, cuda_install_decision, dependency_status, recovery_command
from app.doctor import run_doctor
from app.main import ensure_dependencies, warn_runtime_dependency_fallbacks
from app.results_writer import runtime_environment


def test_dependency_status_includes_descriptions_and_repair_commands():
    status = dependency_status()

    assert DEFAULT_CONFIG["runtime"]["prefer_gpu"] is True
    assert DEFAULT_CONFIG["dependency_install"]["allow_cuda_install"] is True
    assert DEFAULT_CONFIG["dependency_install"]["prefer_cpu_safe_defaults"] is False
    assert status["onnx"]["description"]
    assert status["onnx"]["requirement_file"] == "requirements/onnx.txt"
    assert "pip install -r requirements/onnx.txt" in status["onnx"]["recovery_command"]
    assert "requirements/onnx_cuda.txt" in status["onnx"]["cuda_recovery_command"]
    assert recovery_command("llama_cpp").endswith("requirements/llama_cpp.txt")
    assert "requirements/torch_cuda_cu128.txt" in recovery_command("transformers_cpu", use_cuda=True)
    assert "requirements/faster_whisper_cuda.txt" in recovery_command("faster_whisper", use_cuda=True)
    assert "requirements/llama_cpp_cuda_cu125.txt" in recovery_command("llama_cpp", use_cuda=True)
    assert "requirements/openai_whisper.txt" in recovery_command("openai_whisper", use_cuda=True)


def test_gpu_capable_runtime_groups_have_cuda_overrides():
    expected = {"onnx", "transformers_cpu", "openai_whisper", "faster_whisper", "llama_cpp"}

    assert expected <= set(CUDA_INSTALL_OVERRIDES)
    assert "whisper_cpp" not in CUDA_INSTALL_OVERRIDES
    assert ("onnx", "directml") in ACCELERATOR_OVERRIDES
    assert ("onnx", "openvino") in ACCELERATOR_OVERRIDES
    assert ("llama_cpp", "vulkan") in ACCELERATOR_OVERRIDES


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
    assert decision["requirement_files"] == ["requirements/llama_cpp_vulkan.txt"]


def test_llama_does_not_offer_vulkan_without_sdk(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: False)
    monkeypatch.setattr("app.dependency_manager.vulkan_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager.vulkan_sdk_detected", lambda: False)

    decision = acceleration_install_decision(
        {"runtime": {"provider": "auto"}, "dependency_install": {"allow_accelerator_install": True, "allow_cuda_install": True}},
        "llama_cpp",
    )

    assert decision["use_accelerator"] is False
    assert decision["accelerator"] is None


def test_explicit_vulkan_provider_requires_sdk(monkeypatch):
    monkeypatch.setattr("app.dependency_manager.vulkan_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager.vulkan_sdk_detected", lambda: False)

    decision = acceleration_install_decision(
        {"runtime": {"provider": "vulkan"}, "dependency_install": {"allow_accelerator_install": True}},
        "llama_cpp",
    )

    assert decision["use_accelerator"] is False
    assert decision["accelerator"] == "vulkan"
    assert "Vulkan SDK" in decision["reason"]


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
    assert "CPUExecutionProvider" in output


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
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    monkeypatch.setattr("app.dependency_manager.missing_modules", lambda group: ["onnxruntime"] if group == "onnx" else [])
    monkeypatch.setattr("app.dependency_manager.install_group_for_config", lambda group, root, config: (_ for _ in ()).throw(RuntimeError("pip failed")))

    kept, _ = ensure_dependencies([good, bad], {"dependency_install": {"auto_install_missing_runtime_dependencies": True}})

    output = capsys.readouterr().out
    assert kept == [good]
    assert "Install failed for onnx" in output
    assert "Manual repair command:" in output
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
    monkeypatch.setattr("app.dependency_manager.missing_modules", lambda group: ["onnxruntime"] if group == "onnx" else [])

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
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    monkeypatch.setattr("app.dependency_manager.missing_modules", lambda group: ["llama_cpp"] if group == "llama_cpp" else [])
    monkeypatch.setattr("app.dependency_manager.install_group_for_config", lambda group, root, config: (_ for _ in ()).throw(RuntimeError("pip failed")))

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
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager.onnx_provider_available", lambda provider: not missing["before"])

    def fake_missing(group):
        if group == "onnx" and missing["before"]:
            return ["onnxruntime"]
        return []

    def fake_install(group, root, config):
        decision = acceleration_install_decision(config, group)
        calls.append((group, decision["use_accelerator"], decision["accelerator"]))
        missing["before"] = False

    monkeypatch.setattr("app.dependency_manager.missing_modules", fake_missing)
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


def test_cuda_repair_triggers_for_cpu_only_torch(monkeypatch):
    from app.dependency_manager import missing_modules_for_config

    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager.missing_modules", lambda group: [])
    monkeypatch.setattr("app.dependency_manager.torch_cuda_available", lambda: False)

    missing = missing_modules_for_config(
        "transformers_cpu",
        {
            "runtime": {"provider": "auto", "prefer_gpu": True},
            "dependency_install": {"allow_cuda_install": True},
        },
    )

    assert missing == ["torch CUDA wheel"]


def test_cuda_repair_triggers_for_llama_cpp_without_gpu_offload(monkeypatch):
    from app.dependency_manager import missing_modules_for_config

    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: True)
    monkeypatch.setattr("app.dependency_manager.missing_modules", lambda group: [])
    monkeypatch.setattr("app.dependency_manager.llama_cpp_cuda_capable", lambda: False)

    missing = missing_modules_for_config(
        "llama_cpp",
        {
            "runtime": {"provider": "auto", "prefer_gpu": True},
            "dependency_install": {"allow_cuda_install": True},
        },
    )

    assert missing == ["llama-cpp-python GPU offload build"]


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
