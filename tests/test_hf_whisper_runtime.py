from pathlib import Path
import sys
from types import SimpleNamespace

from app.adapters.base import ModelCandidate
from app.adapters.hf_whisper_asr import HFWhisperASRAdapter
from app.runtime_plan import ResolvedRuntimePlan


def candidate(tmp_path: Path) -> ModelCandidate:
    return ModelCandidate(
        candidate_id="whisper",
        display_name="Whisper",
        family_name="Whisper",
        backend="transformers",
        container_format="safetensors",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="32-bit / FP32",
        path=tmp_path,
        adapter_name="hf_whisper_asr",
        runnable=True,
    )


def test_hf_whisper_retries_cpu_after_cuda_pipeline_failure(monkeypatch, tmp_path: Path):
    calls = []

    class FakeTorch:
        float16 = "float16"

    def fake_pipeline(*args, **kwargs):
        calls.append(kwargs["device"])
        if kwargs["device"] == 0:
            raise RuntimeError("cuda failed")
        return lambda item, **call_kwargs: {"text": "ok"}

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=fake_pipeline))
    monkeypatch.setattr(
        "app.adapters.hf_whisper_asr.resolve_runtime_plan",
        lambda family, config: ResolvedRuntimePlan(family, "cuda", "cuda", "cuda", None, True, True, "verified"),
    )

    adapter = HFWhisperASRAdapter().load(candidate(tmp_path), {"provider": "cuda", "prefer_gpu": True})

    assert calls == [0, -1]
    assert adapter.device == "cpu"
    assert adapter.runtime_plan.actual_provider == "cpu"
    assert "retried CPU" in adapter.load_warnings[-1]
