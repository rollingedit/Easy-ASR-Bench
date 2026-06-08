from pathlib import Path
from types import SimpleNamespace

import pytest

from app.html_report_builder import build_html_report


def test_html_has_visual_diff_pagination_balanced_and_timestamp_validation():
    html = build_html_report(
        {
            "source": {"name": "x.wav", "duration_seconds": 1, "sha256": "abc"},
            "chunk_plan": {
                "chunks": [
                    {"chunk_id": f"{index:04d}", "start_seconds": index, "end_seconds": index + 1, "start_timestamp": "00:00:00.000", "end_timestamp": "00:00:01.000"}
                    for index in range(30)
                ]
            },
            "runs": [
                {
                    "model": {"candidate_id": "m1", "display_name": "Model", "precision": "fp32", "backend": "test"},
                    "transcript_chunks": [{"chunk_id": "0000", "text": "hello world"}],
                    "metrics": {"audio_seconds_per_wall_second": 1, "peak_process_memory_mb": 100},
                    "errors": [],
                }
            ],
            "pairwise_differences": {},
        }
    )

    assert "function alignment" in html
    assert "renderAlignment" in html
    assert "pageSize = 25" in html
    assert "Balanced" in html
    assert "Timestamp mismatches" in html


def test_generic_onnx_records_provider_summary(monkeypatch, tmp_path: Path):
    import numpy as np
    import app.adapters.generic_onnx_manifest as generic
    from app.adapters.base import ModelCandidate

    class FakeSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="input_values")]

        def run(self, outputs, feed):
            return [np.array([[[0.1, 0.8], [0.8, 0.1]]], dtype=np.float32)]

    monkeypatch.setattr(generic, "build_manifest_feed", lambda session, manifest, samples: {"input_values": samples[None, :]})
    (tmp_path / "vocab.json").write_text('{"|": 0, "a": 1}', encoding="utf-8")
    adapter = generic.GenericOnnxManifestAdapter()
    adapter.candidate = ModelCandidate(
        candidate_id="m",
        display_name="m",
        family_name="m",
        backend="onnxruntime",
        container_format="onnx",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="fp32",
        path=tmp_path,
        adapter_name=adapter.name,
        runnable=True,
        metadata={
            "manifest": {
                "files": {"model": "model.onnx"},
                "decoding": {"type": "ctc", "blank_token_id": 0, "vocab_file": "vocab.json"},
                "outputs": {"logits": "logits"},
            }
        },
    )
    adapter.session = FakeSession()
    adapter.requested_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    result = adapter.transcribe_chunks([SimpleNamespace(samples=np.zeros(10, dtype=np.float32))], [{"chunk_id": "0001", "start_seconds": 0, "end_seconds": 1}])

    assert result.metrics["provider_summary"]["cuda_requested"] is True
    assert result.metrics["provider_summary"]["cuda_active"] is False
    assert result.metrics["provider_summary"]["provider_fallback"] is True

    adapter.requested_runtime_provider = "openvino"
    adapter.requested_providers = ["CPUExecutionProvider"]
    result = adapter.transcribe_chunks([SimpleNamespace(samples=np.zeros(10, dtype=np.float32))], [{"chunk_id": "0001", "start_seconds": 0, "end_seconds": 1}])

    assert result.metrics["provider_summary"]["requested_runtime_provider"] == "openvino"
    assert result.metrics["provider_summary"]["openvino_requested"] is True
    assert result.metrics["provider_summary"]["openvino_active"] is False
    assert result.metrics["provider_summary"]["provider_fallback"] is True


def test_faster_whisper_cpu_float16_reports_effective_compute(monkeypatch, tmp_path: Path):
    from app.adapters.base import ModelCandidate
    import app.adapters.faster_whisper_asr as module

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, samples, beam_size=1):
            return [], SimpleNamespace(language="en", language_probability=1.0)

    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setattr(module, "probe_faster_whisper_load", lambda path, device, compute_type: "")
    adapter = module.FasterWhisperASRAdapter()
    candidate = ModelCandidate(
        candidate_id="fw",
        display_name="fw",
        family_name="fw",
        backend="faster-whisper",
        container_format="ctranslate2",
        task="automatic-speech-recognition",
        precision="float16",
        quantization_label="fp16",
        path=tmp_path,
        adapter_name=adapter.name,
        runnable=True,
    )

    adapter.load(candidate, {"provider": "cpu"})
    result = adapter.transcribe_chunks([SimpleNamespace(samples=[])], [{"chunk_id": "0001", "start_seconds": 0, "end_seconds": 1}])

    assert result.metrics["requested_compute_type"] == "float16"
    assert result.metrics["effective_compute_type"] == "float32"
    assert result.metrics["device"] == "cpu"
