from __future__ import annotations

import os
from pathlib import Path


def _has_non_ascii(value: str) -> bool:
    return any(ord(char) > 127 for char in value)


def _is_onedrive_path(path: Path, env: dict[str, str]) -> bool:
    parts = {part.lower() for part in path.parts}
    if any("onedrive" in part for part in parts):
        return True
    for key in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        root = env.get(key)
        if not root:
            continue
        try:
            path.resolve().relative_to(Path(root).resolve())
            return True
        except (OSError, ValueError):
            continue
    return False


def _path_record(kind: str, path: Path, env: dict[str, str]) -> dict:
    value = str(path)
    warnings: list[str] = []
    if _has_non_ascii(value):
        warnings.append("non_ascii_path")
    if _is_onedrive_path(path, env):
        warnings.append("onedrive_or_redirected_path")
    return {"kind": kind, "path": value, "warnings": warnings}


def build_path_diagnostics(config: dict, *, project_root: Path | None = None, env: dict[str, str] | None = None) -> dict:
    environment = dict(os.environ if env is None else env)
    root = Path.cwd() if project_root is None else project_root
    records: list[dict] = [_path_record("install_root", root, environment)]
    user_profile = environment.get("USERPROFILE") or environment.get("HOME")
    if user_profile:
        records.append(_path_record("user_profile", Path(user_profile), environment))
    folders = config.get("folders", {})
    if isinstance(folders, dict):
        for name, raw in sorted(folders.items()):
            path = Path(str(raw))
            if not path.is_absolute():
                path = root / path
            records.append(_path_record(f"folder:{name}", path, environment))
    warnings = [
        {
            "kind": record["kind"],
            "path": record["path"],
            "code": code,
            "message": _warning_message(code),
        }
        for record in records
        for code in record["warnings"]
    ]
    return {
        "schema": "easy_asr_bench.path_diagnostics.v1",
        "records": records,
        "warnings": warnings,
        "ok": not warnings,
    }


def _warning_message(code: str) -> str:
    if code == "non_ascii_path":
        return "Path contains non-ASCII characters; some native Windows runtimes and package installers are less reliable in Unicode paths."
    if code == "onedrive_or_redirected_path":
        return "Path appears to be under OneDrive or a redirected profile; native runtimes, large model files, and temp files may be more fragile there."
    return code
