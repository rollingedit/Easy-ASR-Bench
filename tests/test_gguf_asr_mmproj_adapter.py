import json
import sys
import types
from pathlib import Path

import numpy as np

from app.adapters.gguf_asr_mmproj import GGUFASRMMProjAdapter, extract_cli_transcript, runtime_value
from app.media import AudioChunk


def test_gguf_asr_adapter_discovers_complete_mmproj_pair(tmp_path: Path):
    model = tmp_path / "qwen-asr"
    model.mkdir()
    (model / "Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"gguf")
    (model / "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf").write_bytes(b"gguf")

    candidates = GGUFASRMMProjAdapter().discover(tmp_path)

    assert len(candidates) == 1
    assert candidates[0].runnable is True
    assert candidates[0].runnable_after_dependency_install is True
    assert candidates[0].adapter_name == "gguf_asr_mmproj"
    assert candidates[0].metadata["model_status"] == ""
    assert candidates[0].metadata["model_path"].endswith("Qwen3-ASR-1.7B-Q8_0.gguf")
    assert candidates[0].metadata["mmproj_path"].endswith("mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")
    assert candidates[0].dependency_groups == ["llama_cpp", "llama_mtmd"]


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
    assert candidate.runnable_after_dependency_install is True
    assert candidate.metadata["model_status"] == ""
    assert candidate.metadata["model_path"].endswith("Qwen3-ASR-1.7B-Q4_K_M.gguf")


def test_gguf_asr_adapter_uses_manifest_for_real_qwen_quant_mismatched_projector(tmp_path: Path):
    model = tmp_path / "qwen-asr"
    model.mkdir()
    (model / "Qwen3-ASR-0.6B.Q4_K_M.gguf").write_bytes(b"gguf")
    (model / "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf").write_bytes(b"gguf")
    (model / "model_package.json").write_text(
        json.dumps({"schema": "easy_asr_bench.model_package.v1", "artifacts": {"main_model": "Qwen3-ASR-0.6B.Q4_K_M.gguf", "projector": "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf"}}),
        encoding="utf-8",
    )

    candidate = GGUFASRMMProjAdapter().discover(tmp_path)[0]

    assert candidate.missing_files == []
    assert candidate.metadata["model_path"].endswith("Qwen3-ASR-0.6B.Q4_K_M.gguf")
    assert candidate.metadata["mmproj_path"].endswith("Qwen3-ASR-0.6B.mmproj-Q8_0.gguf")


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


def test_extract_cli_transcript_removes_timestamped_mtmd_noise():
    output = """
0.03.597.046 W init_audio: audio input is in experimental stage and may have reduced quality:
    https://github.com/ggml-org/llama.cpp/discussions/13759
0.03.597.173 I main: loading model: model.gguf
0.03.597.187 W       For normal use cases, please use the standard llama-cli
easy asr bench real model smoke test
"""

    assert extract_cli_transcript(output) == "easy asr bench real model smoke test"


def test_runtime_value_accepts_flat_app_config_and_nested_direct_config():
    assert runtime_value({"ar_prompt": "flat"}, "ar_prompt", "default") == "flat"
    assert runtime_value({"transcription": {"ar_prompt": "nested"}}, "ar_prompt", "default") == "nested"
    assert runtime_value({"llama_cpp": {"timeout_seconds": 12}}, "timeout_seconds", 600, "llama_cpp") == 12


def test_required_dependency_groups_separates_text_llm_and_mtmd_runtime():
    assert GGUFASRMMProjAdapter().required_dependency_groups(None) == ["llama_cpp", "llama_mtmd"]


def test_extract_cli_transcript_removes_qwen_asr_language_prefix():
    assert extract_cli_transcript("language English<asr_text>Easy ASR Bench Real Model Smoke Test.") == "Easy ASR Bench Real Model Smoke Test."
