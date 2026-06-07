import json
import sys
import types
from pathlib import Path

import numpy as np

from app.adapters.gguf_asr_mmproj import GGUFASRMMProjAdapter, extract_cli_transcript
from app.media import AudioChunk


def test_gguf_asr_adapter_discovers_complete_mmproj_pair(tmp_path: Path):
    model = tmp_path / "qwen-asr"
    model.mkdir()
    (model / "Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"gguf")
    (model / "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"gguf")

    candidates = GGUFASRMMProjAdapter().discover(tmp_path)

    assert len(candidates) == 1
    assert candidates[0].runnable is True
    assert candidates[0].adapter_name == "gguf_asr_mmproj"
    assert candidates[0].metadata["model_path"].endswith("Qwen3-ASR-1.7B-Q8_0.gguf")
    assert candidates[0].metadata["mmproj_path"].endswith("mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")


def test_gguf_asr_adapter_rejects_mismatched_projector_quant(tmp_path: Path):
    model = tmp_path / "qwen-asr"
    model.mkdir()
    (model / "Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"gguf")
    (model / "mmproj-Qwen3-ASR-1.7B-Q4_K_M.gguf").write_bytes(b"gguf")

    candidate = GGUFASRMMProjAdapter().discover(tmp_path)[0]

    assert candidate.runnable is False
    assert "matching mmproj .gguf" in candidate.missing_files


def test_gguf_asr_adapter_marks_multiple_pairs_ambiguous_without_manifest(tmp_path: Path):
    model = tmp_path / "qwen-asr"
    model.mkdir()
    (model / "Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"gguf")
    (model / "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"gguf")
    (model / "Qwen3-ASR-1.7B-Q4_K_M.gguf").write_bytes(b"gguf")
    (model / "mmproj-Qwen3-ASR-1.7B-Q4_K_M.gguf").write_bytes(b"gguf")

    candidate = GGUFASRMMProjAdapter().discover(tmp_path)[0]

    assert candidate.runnable is False
    assert "model_package.json exact GGUF ASR pairing manifest" in candidate.missing_files


def test_gguf_asr_adapter_uses_manifest_pair_when_multiple_pairs_exist(tmp_path: Path):
    model = tmp_path / "qwen-asr"
    model.mkdir()
    (model / "Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"gguf")
    (model / "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"gguf")
    (model / "Qwen3-ASR-1.7B-Q4_K_M.gguf").write_bytes(b"gguf")
    (model / "mmproj-Qwen3-ASR-1.7B-Q4_K_M.gguf").write_bytes(b"gguf")
    (model / "model_package.json").write_text(
        json.dumps({"schema": "easy_asr_bench.model_package.v1", "artifacts": {"main_model": "Qwen3-ASR-1.7B-Q4_K_M.gguf", "projector": "mmproj-Qwen3-ASR-1.7B-Q4_K_M.gguf"}}),
        encoding="utf-8",
    )

    candidate = GGUFASRMMProjAdapter().discover(tmp_path)[0]

    assert candidate.runnable is True
    assert candidate.metadata["model_path"].endswith("Qwen3-ASR-1.7B-Q4_K_M.gguf")


def test_gguf_asr_python_backend_transcribes_chunks(tmp_path: Path, monkeypatch):
    model = tmp_path / "qwen-asr"
    model.mkdir()
    (model / "Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"gguf")
    (model / "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"gguf")

    llama_cpp = types.ModuleType("llama_cpp")
    llama_chat_format = types.ModuleType("llama_cpp.llama_chat_format")

    class FakeHandler:
        def __init__(self, clip_model_path, verbose=False):
            self.clip_model_path = clip_model_path

    class FakeLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": "hello world"}}]}

    llama_cpp.Llama = FakeLlama
    llama_chat_format.Qwen3ASRChatHandler = FakeHandler
    monkeypatch.setitem(sys.modules, "llama_cpp", llama_cpp)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", llama_chat_format)

    candidate = GGUFASRMMProjAdapter().discover(tmp_path)[0]
    adapter = GGUFASRMMProjAdapter().load(candidate, {"provider": "cpu", "prefer_gpu": False, "transcription": {"ar_prompt": "Transcribe"}})
    result = adapter.transcribe_chunks(
        [AudioChunk(0, 0.0, 1.0, np.zeros(16000, dtype=np.float32))],
        [{"chunk_id": "0001", "start_seconds": 0.0, "end_seconds": 1.0}],
    )

    assert adapter.backend == "llama-cpp-python-qwen3-asr"
    assert result.transcript_chunks[0].text == "hello world"
    assert result.metrics["provider"] == "llama-cpp-python-qwen3-asr"


def test_extract_cli_transcript_removes_common_runtime_noise():
    output = """
llama_init: loading model
system_info: cpu
actual transcript
"""

    assert extract_cli_transcript(output) == "actual transcript"
