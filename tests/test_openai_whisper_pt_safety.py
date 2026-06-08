import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from app.media import AudioChunk
import app.adapters.openai_whisper_pt as openai_pt
from app.adapters.openai_whisper_pt import OpenAIWhisperPTAdapter


def test_official_openai_whisper_allowlist_is_populated():
    assert openai_pt.KNOWN_OFFICIAL_SHA256["tiny.pt"] == "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9"
    assert openai_pt.KNOWN_OFFICIAL_SHA256["large-v3.pt"] == "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb"
    assert "large-v3-turbo.pt" in openai_pt.KNOWN_OFFICIAL_SHA256


def write_checkpoint(path: Path, data: bytes = b"checkpoint") -> str:
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_known_official_hash_is_runnable(tmp_path: Path, monkeypatch):
    path = tmp_path / "tiny.pt"
    digest = write_checkpoint(path)
    monkeypatch.setattr(openai_pt, "KNOWN_OFFICIAL_SHA256", {"tiny.pt": digest})

    candidates = OpenAIWhisperPTAdapter().discover(tmp_path)

    assert candidates[0].runnable is True
    assert candidates[0].warnings == []


def test_known_filename_wrong_hash_is_blocked(tmp_path: Path, monkeypatch):
    path = tmp_path / "tiny.pt"
    write_checkpoint(path)
    monkeypatch.setattr(openai_pt, "KNOWN_OFFICIAL_SHA256", {"tiny.pt": "0" * 64})

    candidates = OpenAIWhisperPTAdapter().discover(tmp_path)

    assert candidates[0].runnable is False
    assert any("filenames are not trusted" in warning for warning in candidates[0].warnings)


def test_unknown_pt_is_blocked(tmp_path: Path, monkeypatch):
    write_checkpoint(tmp_path / "custom.pt")
    monkeypatch.setattr(openai_pt, "KNOWN_OFFICIAL_SHA256", {})

    candidates = OpenAIWhisperPTAdapter().discover(tmp_path)

    assert candidates[0].runnable is False
    assert "pickle" in candidates[0].warnings[0]


def test_unsafe_trusted_config_allows_load_with_warning_path(tmp_path: Path, monkeypatch):
    path = tmp_path / "custom.pt"
    write_checkpoint(path)
    monkeypatch.setattr(openai_pt, "KNOWN_OFFICIAL_SHA256", {})

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    whisper = types.ModuleType("whisper")
    loaded = {}

    def fake_load_model(model_path, device=None):
        loaded["path"] = model_path
        loaded["device"] = device
        return types.SimpleNamespace(device="cpu", transcribe=lambda samples, fp16=False: {"text": "hello"})

    whisper.load_model = fake_load_model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "whisper", whisper)
    monkeypatch.setattr(openai_pt, "is_verified_official_checkpoint", lambda p: False)

    candidate = OpenAIWhisperPTAdapter().discover(tmp_path)[0]
    adapter = OpenAIWhisperPTAdapter().load(candidate, {"security": {"allow_pickle_or_pt_files": True}})

    assert adapter.model is not None
    assert loaded["path"] == str(path)
    result = adapter.transcribe_chunks(
        [AudioChunk(0, 0.0, 1.0, np.zeros(16000, dtype=np.float32))],
        [{"chunk_id": "0001", "start_seconds": 0.0, "end_seconds": 1.0}],
    )
    assert result.metrics["unsafe_pt_loading_enabled"] is True
    assert result.metrics["checkpoint_sha256_verified"] is False


def test_openai_whisper_cuda_request_records_cpu_fallback(tmp_path: Path, monkeypatch):
    path = tmp_path / "tiny.pt"
    digest = write_checkpoint(path)
    monkeypatch.setattr(openai_pt, "KNOWN_OFFICIAL_SHA256", {"tiny.pt": digest})

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    whisper = types.ModuleType("whisper")

    def fake_load_model(model_path, device=None):
        return types.SimpleNamespace(device=device or "cpu", transcribe=lambda samples, fp16=False: {"text": "hello"})

    whisper.load_model = fake_load_model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "whisper", whisper)

    candidate = OpenAIWhisperPTAdapter().discover(tmp_path)[0]
    adapter = OpenAIWhisperPTAdapter().load(candidate, {"provider": "cuda", "prefer_gpu": True, "fallback_to_cpu": True})
    result = adapter.transcribe_chunks(
        [AudioChunk(0, 0.0, 1.0, np.zeros(16000, dtype=np.float32))],
        [{"chunk_id": "0001", "start_seconds": 0.0, "end_seconds": 1.0}],
    )

    assert result.metrics["device"] == "cpu"
    assert result.metrics["provider_summary"]["requested_provider"] == "cuda"
    assert result.metrics["provider_summary"]["actual_provider"] == "cpu"
    assert result.metrics["provider_summary"]["fallback_reason"]


def test_unsafe_pt_load_is_rejected_without_config(tmp_path: Path, monkeypatch):
    path = tmp_path / "custom.pt"
    write_checkpoint(path)
    monkeypatch.setattr(openai_pt, "KNOWN_OFFICIAL_SHA256", {})
    candidate = OpenAIWhisperPTAdapter().discover(tmp_path)[0]

    with pytest.raises(RuntimeError, match="Blocked .pt checkpoint"):
        OpenAIWhisperPTAdapter().load(candidate, {"security": {"allow_pickle_or_pt_files": False}})
