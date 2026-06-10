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
class AudioSamples:
    path: Path
    sample_count: int
    sample_rate: int = 16000

    def __len__(self) -> int:
        return self.sample_count


@dataclass(frozen=True, init=False)
class AudioChunk:
    index: int
    start_seconds: float
    end_seconds: float
    _samples: np.ndarray | None
    cut_reason: str = "end_of_audio"
    rms_db: float = -120.0
    source_path: Path | None = None
    start_sample: int = 0
    end_sample: int = 0
    sample_rate: int = 16000

    def __init__(
        self,
        index: int,
        start_seconds: float,
        end_seconds: float,
        samples: np.ndarray | None = None,
        cut_reason: str = "end_of_audio",
        rms_db: float = -120.0,
        *,
        source_path: Path | None = None,
        start_sample: int | None = None,
        end_sample: int | None = None,
        sample_rate: int = 16000,
    ) -> None:
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "start_seconds", start_seconds)
        object.__setattr__(self, "end_seconds", end_seconds)
        object.__setattr__(self, "_samples", None if samples is None else np.asarray(samples, dtype=np.float32))
        object.__setattr__(self, "cut_reason", cut_reason)
        object.__setattr__(self, "rms_db", rms_db)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "start_sample", int(start_sample if start_sample is not None else round(start_seconds * sample_rate)))
        object.__setattr__(self, "end_sample", int(end_sample if end_sample is not None else round(end_seconds * sample_rate)))
        object.__setattr__(self, "sample_rate", sample_rate)

    @property
    def samples(self) -> np.ndarray:
        if self._samples is not None:
            return self._samples
        if self.source_path is None:
            return np.zeros(0, dtype=np.float32)
        return read_wav_float32_range(self.source_path, self.start_sample, self.end_sample)

    @property
    def materialized_in_memory(self) -> bool:
        return self._samples is not None


@dataclass(frozen=True)
class MediaProbeResult:
    ok: bool
    has_audio: bool
    ffprobe_available: bool
    probe_method: str = "ffprobe"
    error: str | None = None
    raw_stdout: str = ""
    raw_stderr: str = ""


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def ffprobe_exe() -> str:
    ffmpeg = Path(ffmpeg_exe())
    candidate = ffmpeg.with_name("ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe")
    return str(candidate if candidate.exists() else "ffprobe")


def media_timeouts(config: dict | None = None) -> dict[str, float]:
    media = (config or {}).get("media", {}) if isinstance(config, dict) else {}
    return {
        "probe": float(media.get("probe_timeout_seconds", 30)),
        "conversion": float(media.get("conversion_timeout_seconds", 3600)),
    }


def probe_audio_stream(input_path: Path, timeout_seconds: float = 30) -> MediaProbeResult:
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
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return MediaProbeResult(
            ok=False,
            has_audio=True,
            ffprobe_available=True,
            probe_method="ffprobe",
            error=f"ffprobe timed out after {timeout_seconds:g} seconds while inspecting {input_path}",
            raw_stdout=str(exc.stdout or ""),
            raw_stderr=str(exc.stderr or ""),
        )
    except OSError as exc:
        fallback = probe_audio_stream_with_ffmpeg(input_path, timeout_seconds)
        if fallback.ok:
            return fallback
        return MediaProbeResult(
            ok=False,
            has_audio=fallback.has_audio,
            ffprobe_available=False,
            probe_method="ffmpeg",
            error=(
                "ffprobe was not available and ffmpeg fallback could not inspect audio streams: "
                f"{exc}; {fallback.error or 'no ffmpeg detail was captured'}"
            ),
            raw_stdout=fallback.raw_stdout,
            raw_stderr=fallback.raw_stderr,
        )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        return MediaProbeResult(False, True, True, "ffprobe", stderr or "ffprobe could not inspect this file.", completed.stdout, completed.stderr)
    return MediaProbeResult(True, bool(completed.stdout.strip()), True, "ffprobe", None, completed.stdout, completed.stderr)


def probe_audio_stream_with_ffmpeg(input_path: Path, timeout_seconds: float = 30) -> MediaProbeResult:
    command = [ffmpeg_exe(), "-hide_banner", "-i", str(input_path), "-f", "null", "-"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return MediaProbeResult(
            False,
            True,
            False,
            "ffmpeg",
            f"ffmpeg stream probe timed out after {timeout_seconds:g} seconds while inspecting {input_path}",
            str(exc.stdout or ""),
            str(exc.stderr or ""),
        )
    except OSError as exc:
        return MediaProbeResult(False, True, False, "ffmpeg", f"ffmpeg was not available: {exc}")
    combined = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    if not combined.strip():
        return MediaProbeResult(False, True, False, "ffmpeg", "ffmpeg produced no stream information.", completed.stdout, completed.stderr)
    lower = combined.lower()
    has_audio = any("stream #" in line and "audio:" in line for line in lower.splitlines())
    invalid_or_unreadable = any(
        marker in lower
        for marker in (
            "invalid data found",
            "moov atom not found",
            "error opening input",
            "could not find codec parameters",
            "no such file or directory",
        )
    )
    if invalid_or_unreadable and not has_audio:
        return MediaProbeResult(False, True, False, "ffmpeg", combined.strip()[-1200:], completed.stdout, completed.stderr)
    return MediaProbeResult(True, has_audio, False, "ffmpeg", None, completed.stdout, completed.stderr)


def has_audio_stream(input_path: Path) -> bool:
    return probe_audio_stream(input_path).has_audio


def normalize_to_wav(input_path: Path, temp_dir: Path, config: dict | None = None) -> Path:
    timeouts = media_timeouts(config)
    temp_dir.mkdir(parents=True, exist_ok=True)
    if input_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}:
        probe = probe_audio_stream(input_path, timeouts["probe"])
        if probe.ok and not probe.has_audio:
            raise RuntimeError(f"No audio stream found in video: {input_path}")
        if not probe.ok:
            detail = f" {probe.error}" if probe.error else ""
            raise RuntimeError(f"Could not inspect audio streams before conversion: {input_path}.{detail}")
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
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeouts["conversion"])
    except subprocess.TimeoutExpired as exc:
        stderr = str(exc.stderr or "").strip()
        if len(stderr) > 1200:
            stderr = stderr[-1200:]
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(f"FFmpeg conversion timed out after {timeouts['conversion']:g} seconds for {input_path}{detail}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if len(stderr) > 1200:
            stderr = stderr[-1200:]
        raise RuntimeError(f"FFmpeg conversion failed for {input_path}: {stderr or 'no stderr was captured'}")
    return output


def load_wav_float32(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
        sr = 16000
    return np.asarray(audio, dtype=np.float32), sr


def read_wav_float32_range(path: Path, start_sample: int, end_sample: int) -> np.ndarray:
    start = max(0, int(start_sample))
    frames = max(0, int(end_sample) - start)
    if frames <= 0:
        return np.zeros(0, dtype=np.float32)
    with sf.SoundFile(path) as handle:
        handle.seek(start)
        audio = handle.read(frames, dtype="float32", always_2d=False)
        sr = int(handle.samplerate)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def audio_duration_seconds(samples: np.ndarray | AudioSamples, sr: int = 16000) -> float:
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


def _db_rms_wav(path: Path, start: int, end: int) -> float:
    start = max(0, int(start))
    end = max(start, int(end))
    total_square = 0.0
    total_count = 0
    with sf.SoundFile(path) as handle:
        handle.seek(start)
        remaining = end - start
        while remaining > 0:
            frames = min(remaining, 1_048_576)
            audio = handle.read(frames, dtype="float32", always_2d=False)
            if len(audio) == 0:
                break
            if getattr(audio, "ndim", 1) > 1:
                audio = np.mean(audio, axis=1, dtype=np.float32)
            total_square += float(np.sum(np.square(audio, dtype=np.float64)))
            total_count += int(len(audio))
            remaining -= int(len(audio))
    if total_count == 0:
        return -120.0
    rms = float(np.sqrt(total_square / float(total_count)))
    return 20.0 * np.log10(max(rms, 1e-8))


def _best_boundary_wav(
    path: Path,
    total_samples: int,
    target_sample: int,
    search_samples: int,
    window_samples: int,
    silence_threshold_db: float,
) -> tuple[int, str, float]:
    start = max(window_samples, target_sample - search_samples)
    end = min(total_samples - window_samples, target_sample + search_samples)
    if end <= start:
        boundary = min(max(target_sample, 1), total_samples - 1)
        return boundary, "hard_max", _db_rms_wav(path, boundary - window_samples // 2, boundary + window_samples // 2)
    step = max(1, window_samples // 2)
    best_index = target_sample
    best_db = 999.0
    for index in range(start, end + 1, step):
        db = _db_rms_wav(path, index - window_samples // 2, index + window_samples // 2)
        if db < best_db:
            best_db = db
            best_index = index
            if db <= silence_threshold_db:
                break
    reason = "silence" if best_db <= silence_threshold_db else "rms_fallback"
    return min(max(best_index, 1), total_samples - 1), reason, best_db


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


def plan_wav_chunks(wav_path: Path, config: dict, sr: int = 16000) -> list[AudioChunk]:
    chunking = config["chunking"]
    target_seconds = float(chunking["target_chunk_seconds"])
    hard_max_seconds = float(chunking["hard_max_chunk_seconds"])
    search_seconds = float(chunking["boundary_search_seconds"])
    silence_threshold = float(chunking["silence_threshold_db"])
    rms_window_ms = float(chunking["rms_fallback_window_ms"])
    allow_overlap = bool(chunking.get("allow_overlap", False))

    if allow_overlap:
        raise ValueError("allow_overlap=true is not implemented because fair comparison requires exact non-overlapping chunks")

    with sf.SoundFile(wav_path) as handle:
        total_samples = int(handle.frames)
        actual_sr = int(handle.samplerate)
    if actual_sr != sr:
        raise ValueError(f"Expected normalized WAV sample rate {sr}, got {actual_sr}: {wav_path}")

    total_seconds = float(total_samples) / float(sr)
    if total_seconds <= max(1.0, hard_max_seconds):
        return [
            AudioChunk(
                0,
                0.0,
                total_seconds,
                None,
                "end_of_audio",
                _db_rms_wav(wav_path, 0, total_samples),
                source_path=wav_path,
                start_sample=0,
                end_sample=total_samples,
                sample_rate=sr,
            )
        ]

    target_samples = int(target_seconds * sr)
    hard_max_samples = int(hard_max_seconds * sr)
    search_samples = int(search_seconds * sr)
    window_samples = max(1, int((rms_window_ms / 1000.0) * sr))

    chunks: list[AudioChunk] = []
    start = 0
    while start < total_samples:
        remaining = total_samples - start
        cut_reason = "end_of_audio"
        rms_db = _db_rms_wav(wav_path, start, total_samples)
        if remaining <= hard_max_samples:
            end = total_samples
        else:
            target = start + min(target_samples, hard_max_samples)
            end, cut_reason, rms_db = _best_boundary_wav(wav_path, total_samples, target, search_samples, window_samples, silence_threshold)
            if end <= start:
                end = min(start + hard_max_samples, total_samples)
                cut_reason = "hard_max"
                rms_db = _db_rms_wav(wav_path, end - window_samples // 2, end + window_samples // 2)
            elif end - start >= hard_max_samples:
                cut_reason = "hard_max"
        chunks.append(
            AudioChunk(
                index=len(chunks),
                start_seconds=start / sr,
                end_seconds=end / sr,
                samples=None,
                cut_reason=cut_reason,
                rms_db=float(rms_db),
                source_path=wav_path,
                start_sample=start,
                end_sample=end,
                sample_rate=sr,
            )
        )
        start = end
    return chunks


def prepare_audio(input_path: Path, temp_dir: Path, config: dict) -> tuple[Path, AudioSamples, list[AudioChunk]]:
    wav_path = normalize_to_wav(input_path, temp_dir, config)
    with sf.SoundFile(wav_path) as handle:
        sample_count = int(handle.frames)
        sr = int(handle.samplerate)
    samples = AudioSamples(wav_path, sample_count, sr)
    chunks = plan_wav_chunks(wav_path, config, sr)
    return wav_path, samples, chunks
