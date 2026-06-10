import numpy as np
from pathlib import Path

from app.adapters.base import ModelCandidate
from app.adapters.granite_onnx_ar import GraniteOnnxARAdapter
from app.adapters.granite_onnx_nar import GraniteOnnxNARAdapter


class FailingRunner:
    actual_provider = "CPUExecutionProvider"
    model_load_seconds = 0.0

    def transcribe_array(self, samples, *args):
        raise RuntimeError("chunk boom")


def candidate(adapter_name: str) -> ModelCandidate:
    return ModelCandidate(
        candidate_id=f"{adapter_name}_candidate",
        display_name="Adapter Candidate",
        family_name="Family",
        backend="onnxruntime",
        container_format="onnx",
        task="automatic-speech-recognition",
        precision="unknown",
        quantization_label="Unknown precision",
        path=Path("."),
        adapter_name=adapter_name,
        runnable=True,
    )


def chunk_data():
    return [type("Chunk", (), {"samples": np.zeros(160, dtype=np.float32)})()], [{"chunk_id": "chunk-0", "start_seconds": 0.0, "end_seconds": 0.01}]


def test_granite_ar_chunk_errors_are_reported():
    adapter = GraniteOnnxARAdapter()
    adapter.runner = FailingRunner()
    adapter.candidate = candidate(adapter.name)
    adapter.runtime_config = {"ar_prompt": "transcribe"}
    chunks, metadata = chunk_data()

    result = adapter.transcribe_chunks(chunks, metadata)

    assert result.errors[0]["status"] == "chunk_failed"
    assert result.errors[0]["chunk_id"] == "chunk-0"
    assert result.errors[0]["message"] == "chunk boom"
    assert result.transcript_chunks[0].text == ""


def test_granite_nar_chunk_errors_are_reported():
    adapter = GraniteOnnxNARAdapter()
    adapter.runner = FailingRunner()
    adapter.candidate = candidate(adapter.name)
    chunks, metadata = chunk_data()

    result = adapter.transcribe_chunks(chunks, metadata)

    assert result.errors[0]["status"] == "chunk_failed"
    assert result.errors[0]["chunk_id"] == "chunk-0"
    assert result.errors[0]["message"] == "chunk boom"
    assert result.transcript_chunks[0].text == ""
