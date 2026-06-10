from __future__ import annotations

import time
from pathlib import Path


TEMP_WAV_SUFFIX = "_16k_mono.wav"


def sweep_stale_temp_wavs(config: dict, *, now: float | None = None) -> dict:
    advanced = config.get("advanced", {})
    temp_root = Path(str(config.get("folders", {}).get("temp", "Temp")))
    age_hours = float(advanced.get("stale_temp_wav_hours", 24))
    cutoff = (time.time() if now is None else now) - max(0.0, age_hours) * 3600
    report = {
        "schema": "easy_asr_bench.temp_cleanup.v1",
        "temp_root": str(temp_root),
        "enabled": not bool(advanced.get("keep_temp_wavs", False)),
        "stale_after_hours": age_hours,
        "removed": [],
        "preserved": [],
        "failed": [],
        "summary": {"removed": 0, "preserved": 0, "failed": 0},
    }
    if not report["enabled"] or not temp_root.exists():
        return report
    for path in sorted(temp_root.rglob(f"*{TEMP_WAV_SUFFIX}")):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError as exc:
            report["failed"].append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if stat.st_mtime > cutoff:
            report["preserved"].append({"path": str(path), "reason": "recent_or_active"})
            continue
        try:
            path.unlink()
            report["removed"].append({"path": str(path), "bytes": stat.st_size})
        except OSError as exc:
            report["failed"].append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    report["summary"] = {
        "removed": len(report["removed"]),
        "preserved": len(report["preserved"]),
        "failed": len(report["failed"]),
    }
    return report
