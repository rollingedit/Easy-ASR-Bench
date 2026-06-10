from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiskSpaceCheck:
    path: Path
    required_bytes: int | None
    free_bytes: int | None
    ok: bool
    reason: str


def format_bytes(size: int | None) -> str:
    if size is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"


def existing_disk_root(path: Path) -> Path:
    candidate = Path(path)
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


def free_bytes(path: Path) -> int | None:
    try:
        return int(shutil.disk_usage(existing_disk_root(path)).free)
    except OSError:
        return None


def check_disk_space(path: Path, required_bytes: int | None) -> DiskSpaceCheck:
    root = existing_disk_root(path)
    available = free_bytes(root)
    if required_bytes is None:
        return DiskSpaceCheck(root, None, available, True, "size_unknown")
    if available is None:
        return DiskSpaceCheck(root, required_bytes, None, True, "free_space_unknown")
    if available >= required_bytes:
        return DiskSpaceCheck(root, required_bytes, available, True, "enough_space")
    return DiskSpaceCheck(root, required_bytes, available, False, "low_space")


def require_disk_space(path: Path, required_bytes: int | None, *, label: str, allow_low_space: bool = False) -> DiskSpaceCheck:
    check = check_disk_space(path, required_bytes)
    if check.ok or allow_low_space:
        return check
    raise RuntimeError(
        f"Not enough free disk space for {label}: need about {format_bytes(check.required_bytes)}, "
        f"but only {format_bytes(check.free_bytes)} is free on {check.path}. "
        "Free space or set dependency_install.allow_low_disk_space_install=true to override."
    )


def cache_plus_destination_bytes(payload_bytes: int | None) -> int | None:
    if payload_bytes is None:
        return None
    return int(payload_bytes) * 2
