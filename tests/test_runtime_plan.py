from app.runtime_plan import HardwareInfo, resolve_runtime_plan


def test_cpu_runtime_plan_does_not_probe_hardware(monkeypatch):
    def fail_probe():
        raise AssertionError("CPU-only runtime planning should not import GPU/provider diagnostics")

    monkeypatch.setattr("app.runtime_plan.hardware_from_dependency_manager", fail_probe)

    plan = resolve_runtime_plan("faster_whisper", {"provider": "cpu", "prefer_gpu": False})

    assert plan.actual_provider == "cpu"
    assert plan.backend_verified is True
    assert "without GPU dependency probing" in plan.reason


def test_faster_whisper_prefer_gpu_without_verified_cuda_uses_cpu():
    plan = resolve_runtime_plan("faster_whisper", {"provider": "auto", "prefer_gpu": True}, HardwareInfo(nvidia=True, torch_cuda_available=True, ctranslate2_cuda_available=False))

    assert plan.actual_provider == "cpu"
    assert plan.backend_verified is False
    assert "CTranslate2" in (plan.fallback_reason or "")


def test_faster_whisper_uses_cuda_only_when_ctranslate2_backend_verified():
    plan = resolve_runtime_plan("faster_whisper", {"provider": "auto", "prefer_gpu": True}, HardwareInfo(nvidia=True, torch_cuda_available=False, ctranslate2_cuda_available=True))

    assert plan.actual_provider == "cuda"
    assert plan.backend_verified is True


def test_transformers_asr_prefer_gpu_without_torch_cuda_uses_cpu():
    plan = resolve_runtime_plan("transformers_asr", {"provider": "auto", "prefer_gpu": True}, HardwareInfo(nvidia=True, torch_cuda_available=False))

    assert plan.actual_provider == "cpu"
    assert plan.backend_verified is False
    assert "CUDA-enabled Torch" in (plan.fallback_reason or "")


def test_llama_cpp_prefer_gpu_without_backend_uses_cpu():
    plan = resolve_runtime_plan("llama_cpp", {"provider": "auto", "prefer_gpu": True}, HardwareInfo(nvidia=True, llama_cpp_gpu_offload=False))

    assert plan.actual_provider == "cpu"
    assert plan.device == "cpu"
    assert plan.backend_verified is False


def test_llama_cpp_verified_cuda_uses_cuda():
    plan = resolve_runtime_plan("llama_cpp", {"provider": "cuda", "prefer_gpu": True}, HardwareInfo(nvidia=True, llama_cpp_gpu_offload=True))

    assert plan.actual_provider == "cuda"
    assert plan.backend_verified is True


def test_llama_cpp_vulkan_runtime_without_sdk_uses_cpu():
    plan = resolve_runtime_plan("llama_cpp", {"provider": "vulkan"}, HardwareInfo(vulkan_runtime=True, vulkan_sdk=False))

    assert plan.actual_provider == "cpu"
    assert "Vulkan" in plan.reason
