from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start_seconds: float
    end_seconds: float
    samples: np.ndarray
    cut_reason: str = "end_of_audio"
    rms_db: float = -120.0


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def ffprobe_exe() -> str:
    ffmpeg = Path(ffmpeg_exe())
    candidate = ffmpeg.with_name("ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe")
    return str(candidate if candidate.exists() else "ffprobe")


def has_audio_stream(input_path: Path) -> bool:
    command = [
        ffprobe_exe(),
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(input_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
    except OSError:
        return True
    if completed.returncode != 0:
        return True
    return bool(completed.stdout.strip())


def normalize_to_wav(input_path: Path, temp_dir: Path) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    if input_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"} and not has_audio_stream(input_path):
        raise RuntimeError(f"This video has no audio track: {input_path}")
    output = temp_dir / f"{input_path.stem}_{uuid.uuid4().hex[:10]}_16k_mono.wav"
    command = [
        ffmpeg_exe(),
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "FFmpeg conversion failed")
    return output


def load_wav_float32(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
        sr = 16000
    return np.asarray(audio, dtype=np.float32), sr


def audio_duration_seconds(samples: np.ndarray, sr: int = 16000) -> float:
    return float(len(samples)) / float(sr)


def _db_rms(audio: np.ndarray, start: int, end: int) -> float:
    window = audio[max(0, start) : min(len(audio), end)]
    if len(window) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float64)))
    return 20.0 * np.log10(max(rms, 1e-8))


def _best_boundary(
    audio: np.ndarray,
    target_sample: int,
    search_samples: int,
    window_samples: int,
    silence_threshold_db: float,
) -> tuple[int, str, float]:
    start = max(window_samples, target_sample - search_samples)
    end = min(len(audio) - window_samples, target_sample + search_samples)
    if end <= start:
        boundary = min(max(target_sample, 1), len(audio) - 1)
        return boundary, "hard_max", _db_rms(audio, boundary - window_samples // 2, boundary + window_samples // 2)
    step = max(1, window_samples // 2)
    best_index = target_sample
    best_db = 999.0
    for index in range(start, end + 1, step):
        db = _db_rms(audio, index - window_samples // 2, index + window_samples // 2)
        if db < best_db:
            best_db = db
            best_index = index
            if db <= silence_threshold_db:
                break
    reason = "silence" if best_db <= silence_threshold_db else "rms_fallback"
    return min(max(best_index, 1), len(audio) - 1), reason, best_db


def plan_chunks(samples: np.ndarray, config: dict, sr: int = 16000) -> list[AudioChunk]:
    chunking = config["chunking"]
    target_seconds = float(chunking["target_chunk_seconds"])
    hard_max_seconds = float(chunking["hard_max_chunk_seconds"])
    search_seconds = float(chunking["boundary_search_seconds"])
    silence_threshold = float(chunking["silence_threshold_db"])
    rms_window_ms = float(chunking["rms_fallback_window_ms"])
    allow_overlap = bool(chunking.get("allow_overlap", False))

    if allow_overlap:
        raise ValueError("allow_overlap=true is not implemented because fair comparison requires exact non-overlapping chunks")

    total_seconds = audio_duration_seconds(samples, sr)
    if total_seconds <= max(1.0, hard_max_seconds):
        return [AudioChunk(0, 0.0, total_seconds, samples, "end_of_audio", _db_rms(samples, 0, len(samples)))]

    target_samples = int(target_seconds * sr)
    hard_max_samples = int(hard_max_seconds * sr)
    search_samples = int(search_seconds * sr)
    window_samples = max(1, int((rms_window_ms / 1000.0) * sr))

    chunks: list[AudioChunk] = []
    start = 0
    while start < len(samples):
        remaining = len(samples) - start
        cut_reason = "end_of_audio"
        rms_db = _db_rms(samples, start, len(samples))
        if remaining <= hard_max_samples:
            end = len(samples)
        else:
            target = start + min(target_samples, hard_max_samples)
            end, cut_reason, rms_db = _best_boundary(samples, target, search_samples, window_samples, silence_threshold)
            if end <= start:
                end = min(start + hard_max_samples, len(samples))
                cut_reason = "hard_max"
                rms_db = _db_rms(samples, end - window_samples // 2, end + window_samples // 2)
            elif end - start >= hard_max_samples:
                cut_reason = "hard_max"
        chunk_samples = samples[start:end]
        chunks.append(
            AudioChunk(
                index=len(chunks),
                start_seconds=start / sr,
                end_seconds=end / sr,
                samples=np.asarray(chunk_samples, dtype=np.float32),
                cut_reason=cut_reason,
                rms_db=float(rms_db),
            )
        )
        start = end
    return chunks


def prepare_audio(input_path: Path, temp_dir: Path, config: dict) -> tuple[Path, np.ndarray, list[AudioChunk]]:
    wav_path = normalize_to_wav(input_path, temp_dir)
    samples, sr = load_wav_float32(wav_path)
    chunks = plan_chunks(samples, config, sr)
    return wav_path, samples, chunks
