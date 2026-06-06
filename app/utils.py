from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse


def sanitize_windows_drag_drop_path(raw: str) -> Path:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    if text.lower().startswith("file:"):
        parsed = urlparse(text)
        if parsed.scheme.lower() == "file":
            text = unquote(parsed.path)
            if re.match(r"^/[A-Za-z]:/", text):
                text = text[1:]
            text = text.replace("/", "\\")
    return Path(text)


def parse_windows_path_list(raw: str) -> list[Path]:
    text = raw.strip()
    if not text:
        return []
    lexer = shlex.shlex(text, posix=False)
    lexer.whitespace_split = True
    lexer.commenters = ""
    parts = list(lexer)
    return [sanitize_windows_drag_drop_path(part) for part in parts]


def safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return stem or "input"


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def file_key(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"


def sha256_file(path: Path, limit_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    read_total = 0
    with path.open("rb") as handle:
        while True:
            if limit_bytes is None:
                chunk = handle.read(1024 * 1024)
            else:
                remaining = limit_bytes - read_total
                if remaining <= 0:
                    break
                chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            read_total += len(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_stable_file(path: Path, seconds: float) -> None:
    if seconds <= 0:
        return
    previous = (-1, -1.0)
    stable_since = time.monotonic()
    while True:
        stat = path.stat()
        current = (stat.st_size, stat.st_mtime)
        if current == previous:
            if time.monotonic() - stable_since >= seconds:
                return
        else:
            previous = current
            stable_since = time.monotonic()
        time.sleep(0.5)


def read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    partial.replace(path)


def expand_inputs(paths: list[Path], extensions: set[str], recursive: bool, include_skipped: bool = False):
    files: list[Path] = []
    skipped: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)
        elif path.is_file():
            skipped.append(path)
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            for item in iterator:
                if not item.is_file():
                    continue
                if item.suffix.lower() in extensions:
                    files.append(item)
                else:
                    skipped.append(item)
    deduped_files = sorted(dict.fromkeys(files))
    if include_skipped:
        return deduped_files, sorted(dict.fromkeys(skipped))
    return deduped_files
