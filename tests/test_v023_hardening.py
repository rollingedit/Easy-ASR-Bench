import json
from pathlib import Path

import pytest

from app.adapters.base import ModelCandidate
from app.adapters.generic_onnx_manifest import validate_manifest
from app.adapters.openai_whisper_pt import OpenAIWhisperPTAdapter, is_verified_official_checkpoint
from app.model_selector import recommended_candidates
from app.scoring import normalize_words, wer
from app.utils import parse_windows_path_list


def candidate(index: int, family: str, adapter: str, backend: str, precision: str) -> ModelCandidate:
    return ModelCandidate(
        candidate_id=f"c{index}",
        display_name=f"{family} {precision}",
        family_name=family,
        backend=backend,
        container_format="test",
        task="automatic-speech-recognition",
        precision=precision,
        quantization_label=precision,
        path=Path(f"model{index}"),
        adapter_name=adapter,
        runnable=True,
    )


def test_pt_filename_is_not_trusted(tmp_path: Path):
    model = tmp_path / "large-v3.pt"
    model.write_bytes(b"not an official checkpoint")
    runnable, unsupported = OpenAIWhisperPTAdapter().discover(tmp_path), []

    assert runnable[0].runnable is False
    assert is_verified_official_checkpoint(model) is False
    with pytest.raises(RuntimeError):
        OpenAIWhisperPTAdapter().load(runnable[0], {"security": {"allow_pickle_or_pt_files": False}})


def test_generic_onnx_ctc_manifest_requires_vocab(tmp_path: Path):
    (tmp_path / "model.onnx").write_bytes(b"onnx")
    manifest = {
        "schema": "easy_asr_bench.model_manifest.v1",
        "task": "automatic-speech-recognition",
        "files": {"model": "model.onnx"},
        "inputs": {"waveform": {"name": "input_values"}},
        "outputs": {"logits": "logits"},
        "decoding": {"type": "ctc", "blank_token_id": 0},
    }

    assert "decoding.vocab or decoding.vocab_file" in validate_manifest(manifest, tmp_path)


def test_recommended_candidates_are_family_balanced():
    candidates = [
        candidate(1, "granite", "granite_onnx_ar", "onnxruntime", "int8"),
        candidate(2, "granite", "granite_onnx_ar", "onnxruntime", "fp16w"),
        candidate(3, "whisper", "hf_whisper_asr", "transformers", "fp16"),
        candidate(4, "faster-whisper", "faster_whisper", "faster-whisper", "int8"),
    ]

    selected = recommended_candidates(candidates)

    assert len(selected) >= 3
    assert selected != [1, 2]


def test_unicode_normalization_keeps_non_ascii_text():
    assert normalize_words("Café 北京 don't") == ["café", "北京", "don't"]
    assert wer("北京 café", "北京 cafe") < 1.0


def test_parse_multiple_pasted_windows_paths():
    paths = parse_windows_path_list('"C:\\a file.mp3" "D:\\b file.wav" C:\\Folder')

    assert [str(path) for path in paths] == ["C:\\a file.mp3", "D:\\b file.wav", "C:\\Folder"]
