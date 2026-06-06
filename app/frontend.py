from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def input_features(audio_float32_16k_mono: np.ndarray, model_root: Path | None = None) -> np.ndarray:
    """Create Granite Speech encoder features with shape [1, T, 160]."""
    audio = np.asarray(audio_float32_16k_mono, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=80,
        window="hann",
        center=True,
        power=2.0,
    )
    log_mel = np.log10(np.maximum(mel, 1e-10)).astype(np.float32)
    max_value = float(np.max(log_mel)) if log_mel.size else 0.0
    log_mel = np.maximum(log_mel, max_value - 8.0)
    frames = log_mel.T
    if len(frames) % 2 == 1:
        frames = np.concatenate([frames, frames[-1:]], axis=0)
    stacked = frames.reshape(frames.shape[0] // 2, 160)
    return stacked[None, :, :].astype(np.float32)


def validate_against_fixture(model_root: Path) -> dict[str, float | str]:
    fixture_dir = model_root / "test_fixtures"
    expected = fixture_dir / "expected_input_features.npy"
    audio_candidates = list(fixture_dir.glob("*.wav")) + list(fixture_dir.glob("*.flac"))
    if not expected.exists() or not audio_candidates:
        return {"status": "skipped", "reason": "fixture audio or expected_input_features.npy not found"}

    import soundfile as sf

    audio, sr = sf.read(audio_candidates[0], dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
    actual = input_features(audio, model_root)
    expected_array = np.load(expected)
    if actual.shape != expected_array.shape:
        return {"status": "failed", "reason": f"shape mismatch actual={actual.shape} expected={expected_array.shape}"}
    delta = np.abs(actual - expected_array)
    return {
        "status": "ok",
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
    }
