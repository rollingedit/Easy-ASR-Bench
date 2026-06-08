import subprocess
import sys
import types
from pathlib import Path

from app.adapters.base import ModelCandidate
from app.adapters.faster_whisper_asr import FasterWhisperASRAdapter, probe_faster_whisper_load


def candidate(precision: str) -> ModelCandidate:
    return ModelCandidate(
        candidate_id="fw",
        display_name="FW",
        family_name="FW",
        backend="faster-whisper",
        container_format="ctranslate2",
        task="automatic-speech-recognition",
        precision=precision,
        quantization_label=precision,
        path=Path("model"),
        adapter_name="faster_whisper",
        runnable=True,
    )


def test_faster_whisper_cpu_float16_uses_effective_float32(monkeypatch):
    captured = {}

    class WhisperModel:
        def __init__(self, path, device, compute_type):
            captured.update({"path": path, "device": device, "compute_type": compute_type})

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=WhisperModel))
    monkeypatch.setattr("app.adapters.faster_whisper_asr.probe_faster_whisper_load", lambda path, device, compute_type: "")

    adapter = FasterWhisperASRAdapter().load(candidate("fp16"), {"provider": "cpu", "prefer_gpu": False})

    assert captured["device"] == "cpu"
    assert captured["compute_type"] == "float32"
    assert adapter.requested_compute_type == "float16"
    assert adapter.effective_compute_type == "float32"


def test_faster_whisper_native_probe_reports_child_process_crash(monkeypatch):
    def fake_run(command, capture_output=True, text=True, timeout=90):
        return subprocess.CompletedProcess(command, 3221225477, "", "Windows fatal exception: access violation")

    monkeypatch.setattr("app.adapters.faster_whisper_asr.subprocess.run", fake_run)

    error = probe_faster_whisper_load(Path("model"), "cpu", "default")

    assert "native probe process exited" in error
    assert "access violation" in error
