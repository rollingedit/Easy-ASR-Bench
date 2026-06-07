from pathlib import Path
import sys
from types import SimpleNamespace

from app.adapters.base import ModelCandidate
from app.adapters.hf_transformers_asr import HFTransformersASRAdapter
from app.runtime_plan import HardwareInfo, ResolvedRuntimePlan


def candidate(tmp_path: Path) -> ModelCandidate:
    return ModelCandidate(
        candidate_id="hf",
        display_name="HF",
        family_name="HF",
        backend="transformers",
        container_format="safetensors",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="32-bit / FP32",
        path=tmp_path,
        adapter_name="hf_transformers_asr",
        runnable=True,
    )


def test_transformers_asr_load_retries_cpu_after_cuda_failure(monkeypatch, tmp_path: Path):
    calls = []

    def fake_pipeline(*args, **kwargs):
        calls.append(kwargs["device"])
        if kwargs["device"] == 0:
            raise RuntimeError("cuda broken")
        return lambda item: {"text": "ok"}

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=fake_pipeline))
    monkeypatch.setattr(
        "app.adapters.hf_transformers_asr.resolve_runtime_plan",
        lambda family, config: ResolvedRuntimePlan(family, "cuda", "cuda", "cuda", None, True, True, "verified"),
    )

    adapter = HFTransformersASRAdapter().load(candidate(tmp_path), {"provider": "cuda", "prefer_gpu": True})

    assert calls == [0, -1]
    assert adapter.device == "cpu"
    assert adapter.runtime_plan.actual_provider == "cpu"
    assert "retried CPU" in adapter.load_warnings[-1]


def test_transformers_asr_runtime_plan_uses_cpu_without_verified_cuda():
    from app.runtime_plan import resolve_runtime_plan

    plan = resolve_runtime_plan("transformers_asr", {"provider": "cuda", "prefer_gpu": True}, HardwareInfo(nvidia=True, torch_cuda_available=False))

    assert plan.actual_provider == "cpu"
    assert plan.backend_verified is False
