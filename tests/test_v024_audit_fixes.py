import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import app.adapters.generic_onnx_manifest as generic_manifest
from app.adapters.base import ModelCandidate
from app.adapters.generic_onnx_manifest import GenericOnnxManifestAdapter


class FakeSession:
    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [SimpleNamespace(name="input_values")]

    def run(self, outputs, feed):
        del outputs, feed
        return [np.array([[[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]]], dtype=np.float32)]


def test_generic_onnx_transcribe_loads_vocab_from_candidate_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(generic_manifest, "session_input_names", lambda session: ["input_values"])
    (tmp_path / "vocab.json").write_text(json.dumps({"|": 0, "a": 1, "b": 2}), encoding="utf-8")
    manifest = {
        "schema": "easy_asr_bench.model_manifest.v1",
        "task": "automatic-speech-recognition",
        "files": {"model": "model.onnx"},
        "inputs": {"waveform": {"name": "input_values"}},
        "outputs": {"logits": "logits"},
        "preprocessing": {"type": "raw_waveform"},
        "decoding": {"type": "ctc", "blank_token_id": 0, "vocab_file": "vocab.json"},
    }
    adapter = GenericOnnxManifestAdapter()
    adapter.candidate = ModelCandidate(
        candidate_id="generic",
        display_name="generic",
        family_name="generic",
        backend="onnxruntime",
        container_format="onnx",
        task="automatic-speech-recognition",
        precision="unknown",
        quantization_label="unknown",
        path=tmp_path,
        adapter_name=adapter.name,
        runnable=True,
        metadata={"manifest": manifest},
    )
    adapter.session = FakeSession()

    result = adapter.transcribe_chunks([SimpleNamespace(samples=np.zeros(160, dtype=np.float32))], [{"chunk_id": "0001", "start_seconds": 0, "end_seconds": 0.01}])

    assert result.transcript_chunks[0].text == "a"
    assert result.errors == []


def test_installer_preservation_does_not_use_literalpath_wildcard():
    text = Path("installer/install.ps1").read_text(encoding="utf-8")

    assert 'Join-Path $source "*"' not in text
    preservation_block = text.split("function Move-PreservedUserData", 1)[1].split("function Restore-MovedUserData", 1)[0]
    assert "Copy-Item -LiteralPath $child.FullName" not in preservation_block
    assert "Move-PreservedDirectory $source $dest" in preservation_block
    assert 'method = "move_without_model_copy"' in preservation_block
    assert 'status = "moved"' in preservation_block
    assert "Restore-MovedUserData $Preserve $Backup" in text
    assert "[switch]$RemoveUserData" in text
    assert "User data was preserved" in text
