from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from app.media import ffmpeg_exe


REFERENCE_TEXT = "easy asr bench validation audio"


def write_reference_wav(path: Path, seconds: float = 1.5, sample_rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.arange(int(seconds * sample_rate), dtype=np.float32)
    tone = 0.18 * np.sin(2.0 * math.pi * 440.0 * samples / sample_rate)
    sf.write(path, tone.astype(np.float32), sample_rate)
    return path


def _run_ffmpeg(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True, timeout=120)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "ffmpeg failed").strip()
        raise RuntimeError(detail[-1200:])


def write_reference_mp3(path: Path, wav_path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([ffmpeg_exe(), "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", str(path)])
    return path


def write_reference_mp4(path: Path, wav_path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            ffmpeg_exe(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x180:r=25:d=1.5",
            "-i",
            str(wav_path),
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ]
    )
    return path


def write_no_audio_mp4(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            ffmpeg_exe(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x180:r=25:d=1.0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )
    return path


def write_corrupt_media(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a valid media file")
    return path


def create_media_fixture_set(root: Path) -> dict[str, Path]:
    wav = write_reference_wav(root / "reference.wav")
    return {
        "wav": wav,
        "mp3": write_reference_mp3(root / "reference.mp3", wav),
        "mp4": write_reference_mp4(root / "reference.mp4", wav),
        "no_audio_mp4": write_no_audio_mp4(root / "no_audio.mp4"),
        "corrupt": write_corrupt_media(root / "corrupt.mp4"),
    }
