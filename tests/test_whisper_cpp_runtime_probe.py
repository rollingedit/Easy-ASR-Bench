import sys
import types

from app.adapters.whisper_cpp_asr import WhisperCppASRAdapter


def clear_pywhispercpp(monkeypatch):
    monkeypatch.delitem(sys.modules, "pywhispercpp", raising=False)
    monkeypatch.delitem(sys.modules, "pywhispercpp.model", raising=False)


def install_fake_pywhispercpp(monkeypatch, model_cls):
    package = types.ModuleType("pywhispercpp")
    model_module = types.ModuleType("pywhispercpp.model")
    model_module.Model = model_cls
    monkeypatch.setitem(sys.modules, "pywhispercpp", package)
    monkeypatch.setitem(sys.modules, "pywhispercpp.model", model_module)


def test_whisper_cpp_probe_reports_missing_module(monkeypatch):
    clear_pywhispercpp(monkeypatch)

    result = WhisperCppASRAdapter().runtime_probe()

    assert result.ok is False
    assert "not installed" in result.message


def test_whisper_cpp_probe_reports_missing_transcribe(monkeypatch):
    class Model:
        pass

    install_fake_pywhispercpp(monkeypatch, Model)

    result = WhisperCppASRAdapter().runtime_probe()

    assert result.ok is False
    assert "transcribe not found" in result.message


def test_whisper_cpp_probe_accepts_model_with_transcribe(monkeypatch):
    class Model:
        def transcribe(self, samples):
            return []

    install_fake_pywhispercpp(monkeypatch, Model)

    result = WhisperCppASRAdapter().runtime_probe()

    assert result.ok is True
