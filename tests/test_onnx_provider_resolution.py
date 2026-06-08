import importlib
import sys
import types

from app.runtime_plan import HardwareInfo


def load_onnx_common(monkeypatch, providers):
    fake_ort = types.SimpleNamespace(
        get_available_providers=lambda: providers,
        SessionOptions=object,
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        InferenceSession=object,
    )
    fake_tokenizers = types.SimpleNamespace(Tokenizer=object)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tokenizers)
    sys.modules.pop("app.onnx_common", None)
    module = importlib.import_module("app.onnx_common")
    sys.modules.pop("app.onnx_common", None)
    return module


def test_intel_prefers_openvino_before_directml(monkeypatch):
    onnx_common = load_onnx_common(monkeypatch, ["DmlExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"])

    providers = onnx_common.choose_providers("auto", HardwareInfo(intel_gpu=True, windows_gpu=True))

    assert providers[:2] == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]


def test_amd_prefers_directml_before_cpu(monkeypatch):
    onnx_common = load_onnx_common(monkeypatch, ["DmlExecutionProvider", "CPUExecutionProvider"])

    providers = onnx_common.choose_providers("auto", HardwareInfo(amd=True, windows_gpu=True))

    assert providers == ["DmlExecutionProvider", "CPUExecutionProvider"]


def test_nvidia_prefers_cuda_before_cpu(monkeypatch):
    onnx_common = load_onnx_common(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    providers = onnx_common.choose_providers("auto", HardwareInfo(nvidia=True))

    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_missing_requested_provider_falls_back_to_cpu(monkeypatch):
    onnx_common = load_onnx_common(monkeypatch, ["CPUExecutionProvider"])

    assert onnx_common.choose_providers("openvino", HardwareInfo(intel_gpu=True)) == ["CPUExecutionProvider"]
