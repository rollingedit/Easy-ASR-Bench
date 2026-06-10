import json
import os
import time
from pathlib import Path

from app.doctor import build_doctor_report
from app.temp_cleanup import sweep_stale_temp_wavs


def test_temp_cleanup_removes_stale_generated_wavs_and_preserves_recent(tmp_path: Path):
    temp = tmp_path / "Temp"
    temp.mkdir()
    stale = temp / "old_1234567890_16k_mono.wav"
    recent = temp / "new_1234567890_16k_mono.wav"
    unrelated = temp / "keep.wav"
    stale.write_bytes(b"old")
    recent.write_bytes(b"new")
    unrelated.write_bytes(b"keep")
    now = time.time()
    os.utime(stale, (now - 48 * 3600, now - 48 * 3600))
    os.utime(recent, (now, now))

    report = sweep_stale_temp_wavs({"folders": {"temp": str(temp)}, "advanced": {"stale_temp_wav_hours": 24}}, now=now)

    assert report["summary"]["removed"] == 1
    assert report["summary"]["preserved"] == 1
    assert not stale.exists()
    assert recent.exists()
    assert unrelated.exists()


def test_temp_cleanup_respects_keep_temp_wavs(tmp_path: Path):
    temp = tmp_path / "Temp"
    temp.mkdir()
    stale = temp / "old_1234567890_16k_mono.wav"
    stale.write_bytes(b"old")
    now = time.time()
    os.utime(stale, (now - 48 * 3600, now - 48 * 3600))

    report = sweep_stale_temp_wavs({"folders": {"temp": str(temp)}, "advanced": {"keep_temp_wavs": True}}, now=now)

    assert report["enabled"] is False
    assert report["summary"]["removed"] == 0
    assert stale.exists()


def test_doctor_report_includes_temp_cleanup(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "app": {"version": "0.4.0", "version_channel": "prerelease"},
                "folders": {"models": "Models", "input": "Input", "output": "Output", "temp": "Temp", "logs": "Logs", "cache": "Cache"},
                "advanced": {"stale_temp_wav_hours": 24},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    report = build_doctor_report(config_path)

    assert report["temp_cleanup"]["schema"] == "easy_asr_bench.temp_cleanup.v1"
    assert report["temp_cleanup"]["summary"]["removed"] == 0
