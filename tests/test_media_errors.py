import subprocess
from pathlib import Path

import pytest

from app import media


def completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["tool"], returncode, stdout=stdout, stderr=stderr)


def test_video_without_audio_fails_before_conversion(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, capture_output=True, text=True, timeout=None):
        calls.append(command)
        return completed(stdout="", returncode=0)

    monkeypatch.setattr(media, "ffprobe_exe", lambda: "ffprobe")
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="No audio stream found"):
        media.normalize_to_wav(tmp_path / "silent.mp4", tmp_path)

    assert len(calls) == 1
    assert "ffprobe" in calls[0][0].lower()


def test_probe_failure_reports_probe_error_without_conversion(tmp_path: Path, monkeypatch):
    def fake_run(command, capture_output=True, text=True, timeout=None):
        return completed(stderr="invalid data found", returncode=1)

    monkeypatch.setattr(media, "ffprobe_exe", lambda: "ffprobe")
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Could not inspect audio streams"):
        media.normalize_to_wav(tmp_path / "broken.mp4", tmp_path)


def test_ffmpeg_conversion_error_includes_input_path_and_stderr(tmp_path: Path, monkeypatch):
    calls = iter([completed(stdout="0", returncode=0), completed(stderr="codec exploded", returncode=1)])

    def fake_run(command, capture_output=True, text=True, timeout=None):
        return next(calls)

    monkeypatch.setattr(media, "ffprobe_exe", lambda: "ffprobe")
    monkeypatch.setattr(media, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="FFmpeg conversion failed for .*input.mp4.*codec exploded"):
        media.normalize_to_wav(tmp_path / "input.mp4", tmp_path)


def test_ffprobe_missing_falls_back_to_ffmpeg_audio_probe(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []
    output_wav = tmp_path / "input_out_16k_mono.wav"

    def fake_run(command, capture_output=True, text=True, timeout=None):
        calls.append(command)
        if command[0] == "ffprobe":
            raise FileNotFoundError("ffprobe")
        if command[:2] == ["ffmpeg", "-hide_banner"]:
            return completed(stderr="Stream #0:1: Audio: aac, 48000 Hz, stereo", returncode=1)
        output_wav.write_bytes(b"RIFF")
        return completed(returncode=0)

    monkeypatch.setattr(media, "ffprobe_exe", lambda: "ffprobe")
    monkeypatch.setattr(media, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(media.uuid, "uuid4", lambda: type("U", (), {"hex": "out"})())
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    result = media.normalize_to_wav(tmp_path / "input.mp4", tmp_path)

    assert result == output_wav
    assert calls[0][0] == "ffprobe"
    assert calls[1][:2] == ["ffmpeg", "-hide_banner"]
    assert calls[2][0] == "ffmpeg"


def test_ffprobe_missing_ffmpeg_fallback_detects_video_without_audio(tmp_path: Path, monkeypatch):
    def fake_run(command, capture_output=True, text=True, timeout=None):
        if command[0] == "ffprobe":
            raise FileNotFoundError("ffprobe")
        return completed(stderr="Stream #0:0: Video: h264, yuv420p", returncode=1)

    monkeypatch.setattr(media, "ffprobe_exe", lambda: "ffprobe")
    monkeypatch.setattr(media, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="No audio stream found"):
        media.normalize_to_wav(tmp_path / "silent.mp4", tmp_path)


def test_ffprobe_timeout_reports_probe_timeout_without_conversion(tmp_path: Path, monkeypatch):
    def fake_run(command, capture_output=True, text=True, timeout=None):
        raise subprocess.TimeoutExpired(command, timeout or 1, stderr="probe hung")

    monkeypatch.setattr(media, "ffprobe_exe", lambda: "ffprobe")
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="ffprobe timed out after 2 seconds"):
        media.normalize_to_wav(
            tmp_path / "hung.mp4",
            tmp_path,
            {"media": {"probe_timeout_seconds": 2, "conversion_timeout_seconds": 5}},
        )


def test_ffmpeg_conversion_timeout_reports_timeout(tmp_path: Path, monkeypatch):
    calls = iter([completed(stdout="0", returncode=0), subprocess.TimeoutExpired(["ffmpeg"], 3, stderr="convert hung")])

    def fake_run(command, capture_output=True, text=True, timeout=None):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(media, "ffprobe_exe", lambda: "ffprobe")
    monkeypatch.setattr(media, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="FFmpeg conversion timed out after 3 seconds"):
        media.normalize_to_wav(
            tmp_path / "hung.mp4",
            tmp_path,
            {"media": {"probe_timeout_seconds": 2, "conversion_timeout_seconds": 3}},
        )
