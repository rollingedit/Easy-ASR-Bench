import sys
import types
from pathlib import Path

from app.adapters.base import ModelCandidate
from app.adapters.faster_whisper_asr import FasterWhisperASRAdapter


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

    adapter = FasterWhisperASRAdapter().load(candidate("fp16"), {"provider": "cpu", "prefer_gpu": False})

    assert captured["device"] == "cpu"
    assert captured["compute_type"] == "float32"
    assert adapter.requested_compute_type == "float16"
    assert adapter.effective_compute_type == "float32"
