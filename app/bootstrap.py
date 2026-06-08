from __future__ import annotations

from pathlib import Path

from .doctor import run_doctor


def repair(config_path: Path = Path("config.json")) -> int:
    return run_doctor(config_path, repair_all_safe=True)
