import importlib
import sys
import types
from pathlib import Path


def import_with_fake_onnxruntime(monkeypatch):
    fake = types.SimpleNamespace()
    fake.ExecutionMode = types.SimpleNamespace(ORT_SEQUENTIAL="sequential")
    fake.available = ["DmlExecutionProvider", "CPUExecutionProvider"]
    fake.created = {}

    class SessionOptions:
        def __init__(self):
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0
            self.enable_mem_pattern = True
            self.execution_mode = None

    class InferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            fake.created = {"path": path, "sess_options": sess_options, "providers": providers}

    fake.SessionOptions = SessionOptions
    fake.InferenceSession = InferenceSession
    fake.get_available_providers = lambda: list(fake.available)
    fake_tokenizers = types.SimpleNamespace(Tokenizer=object)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tokenizers)
    sys.modules.pop("app.onnx_common", None)
    return importlib.import_module("app.onnx_common"), fake


def test_directml_session_disables_mem_pattern_and_uses_sequential(monkeypatch):
    onnx_common, fake = import_with_fake_onnxruntime(monkeypatch)

    onnx_common.make_session(Path("model.onnx"), ["DmlExecutionProvider", "CPUExecutionProvider"], cpu_threads=2)

    options = fake.created["sess_options"]
    assert fake.created["providers"] == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert options.enable_mem_pattern is False
    assert options.execution_mode == "sequential"
    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 2


def test_openvino_session_adds_openvino_dll_directories(tmp_path, monkeypatch):
    onnx_common, fake = import_with_fake_onnxruntime(monkeypatch)
    package = tmp_path / "openvino"
    libs = package / "libs"
    libs.mkdir(parents=True)
    init_file = package / "__init__.py"
    init_file.write_text("", encoding="utf-8")
    fake_openvino = types.SimpleNamespace(__file__=str(init_file))
    added: list[str] = []

    class Handle:
        pass

    monkeypatch.setitem(sys.modules, "openvino", fake_openvino)
    monkeypatch.setattr(onnx_common.os, "name", "nt")
    monkeypatch.setattr(onnx_common.os, "add_dll_directory", lambda path: added.append(str(path)) or Handle(), raising=False)

    onnx_common.make_session(Path("model.onnx"), ["OpenVINOExecutionProvider", "CPUExecutionProvider"], cpu_threads=0)

    assert fake.created["providers"] == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    assert str(libs) in added
    assert str(package) in added


def test_choose_providers_directml_falls_back_to_cpu_when_unavailable(monkeypatch):
    onnx_common, fake = import_with_fake_onnxruntime(monkeypatch)
    fake.available = ["CPUExecutionProvider"]

    assert onnx_common.choose_providers("directml") == ["CPUExecutionProvider"]
