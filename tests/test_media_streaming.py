from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app import media


def chunking_config() -> dict:
    return {
        "chunking": {
            "target_chunk_seconds": 1.0,
            "hard_max_chunk_seconds": 1.25,
            "boundary_search_seconds": 0.25,
            "silence_threshold_db": -35,
            "rms_fallback_window_ms": 100,
            "allow_overlap": False,
        }
    }


def write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sr)
    return path


def test_prepare_audio_plans_chunks_without_full_wav_load(tmp_path, monkeypatch):
    sr = 16000
    samples = np.concatenate(
        [
            np.full(sr, 0.2, dtype=np.float32),
            np.zeros(sr // 2, dtype=np.float32),
            np.full(sr, 0.2, dtype=np.float32),
        ]
    )
    wav_path = write_wav(tmp_path / "normalized.wav", samples, sr)
    monkeypatch.setattr(media, "normalize_to_wav", lambda input_path, temp_dir: wav_path)
    monkeypatch.setattr(
        media,
        "load_wav_float32",
        lambda path: (_ for _ in ()).throw(AssertionError("prepare_audio must not load the full WAV")),
    )

    normalized, audio_samples, chunks = media.prepare_audio(tmp_path / "source.mp3", tmp_path / "Temp", chunking_config())

    assert normalized == wav_path
    assert len(audio_samples) == len(samples)
    assert len(chunks) >= 2
    assert all(not chunk.materialized_in_memory for chunk in chunks)


def test_lazy_audio_chunk_materializes_only_its_range(tmp_path):
    sr = 16000
    samples = np.linspace(-0.5, 0.5, sr * 2, dtype=np.float32)
    wav_path = write_wav(tmp_path / "normalized.wav", samples, sr)
    chunk = media.AudioChunk(
        0,
        0.25,
        0.75,
        None,
        source_path=wav_path,
        start_sample=sr // 4,
        end_sample=(sr * 3) // 4,
        sample_rate=sr,
    )

    materialized = chunk.samples

    assert not chunk.materialized_in_memory
    assert len(materialized) == sr // 2
    np.testing.assert_allclose(materialized, samples[sr // 4 : (sr * 3) // 4], atol=1e-4)


def test_streaming_chunk_plan_matches_in_memory_boundaries(tmp_path):
    sr = 16000
    tone = np.full(sr, 0.2, dtype=np.float32)
    silence = np.zeros(sr // 2, dtype=np.float32)
    samples = np.concatenate([tone, silence, tone])
    wav_path = write_wav(tmp_path / "normalized.wav", samples, sr)
    eager = media.plan_chunks(samples, chunking_config(), sr)
    streaming = media.plan_wav_chunks(wav_path, chunking_config(), sr)

    assert [(chunk.start_seconds, chunk.end_seconds, chunk.cut_reason) for chunk in streaming] == [
        (chunk.start_seconds, chunk.end_seconds, chunk.cut_reason) for chunk in eager
    ]


def test_plan_wav_chunks_rejects_unexpected_sample_rate(tmp_path):
    wav_path = write_wav(tmp_path / "wrong-rate.wav", np.zeros(8000, dtype=np.float32), 8000)

    with pytest.raises(ValueError, match="Expected normalized WAV sample rate"):
        media.plan_wav_chunks(wav_path, chunking_config(), 16000)
