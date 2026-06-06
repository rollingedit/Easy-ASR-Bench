import subprocess
from pathlib import Path

import pytest

from app import media


def completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["tool"], returncode, stdout=stdout, stderr=stderr)


def test_video_without_audio_fails_before_conversion(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, capture_output=True, text=True):
        calls.append(command)
        return completed(stdout="", returncode=0)

    monkeypatch.setattr(media, "ffprobe_exe", lambda: "ffprobe")
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="No audio stream found"):
        media.normalize_to_wav(tmp_path / "silent.mp4", tmp_path)

    assert len(calls) == 1
    assert "ffprobe" in calls[0][0].lower()


def test_probe_failure_reports_probe_error_without_conversion(tmp_path: Path, monkeypatch):
    def fake_run(command, capture_output=True, text=True):
        return completed(stderr="invalid data found", returncode=1)

    monkeypatch.setattr(media, "ffprobe_exe", lambda: "ffprobe")
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Could not inspect audio streams"):
        media.normalize_to_wav(tmp_path / "broken.mp4", tmp_path)


def test_ffmpeg_conversion_error_includes_input_path_and_stderr(tmp_path: Path, monkeypatch):
    calls = iter([completed(stdout="0", returncode=0), completed(stderr="codec exploded", returncode=1)])

    def fake_run(command, capture_output=True, text=True):
        return next(calls)

    monkeypatch.setattr(media, "ffprobe_exe", lambda: "ffprobe")
    monkeypatch.setattr(media, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="FFmpeg conversion failed for .*input.mp4.*codec exploded"):
        media.normalize_to_wav(tmp_path / "input.mp4", tmp_path)
