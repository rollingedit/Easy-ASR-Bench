from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .utils import expand_inputs, file_key, read_json, sha256_file, wait_for_stable_file, write_json_atomic


@dataclass
class QueueItem:
    source_path: str
    sha256: str
    status: str = "queued"
    output_folder: str = ""
    errors: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class QueueState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = read_json(path, {"schema": "easy_asr_bench.queue_state.v1", "items": []})

    def save(self) -> None:
        write_json_atomic(self.path, self.data)

    def upsert(self, item: QueueItem) -> None:
        items = self.data.setdefault("items", [])
        for existing in items:
            if existing["sha256"] == item.sha256 and existing["source_path"] == item.source_path:
                existing.update(item.__dict__)
                existing["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.save()
                return
        items.append(item.__dict__)
        self.save()

    def mark(self, source_path: Path, status: str, output_folder: str = "", error: str | None = None) -> None:
        source = str(source_path)
        for item in self.data.setdefault("items", []):
            if item["source_path"] == source:
                item["status"] = status
                item["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                if output_folder:
                    item["output_folder"] = output_folder
                if error:
                    item.setdefault("errors", []).append(error)
                self.save()
                return

    def done_hashes(self) -> set[str]:
        return {item["sha256"] for item in self.data.get("items", []) if item.get("status") == "done"}


def discover_queue(paths: list[Path], extensions: set[str], recursive: bool, stability_seconds: float, state: QueueState, skip_done: bool) -> list[Path]:
    files = expand_inputs(paths, extensions, recursive)
    queued: list[Path] = []
    done = state.done_hashes() if skip_done else set()
    for path in files:
        wait_for_stable_file(path, stability_seconds)
        digest = sha256_file(path)
        if digest in done:
            continue
        state.upsert(QueueItem(str(path.resolve()), digest))
        queued.append(path)
    return queued
