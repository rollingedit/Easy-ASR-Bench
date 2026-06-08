from __future__ import annotations

from pathlib import Path

import app.queue_manager as queue_manager
from app.queue_manager import QueueItem, QueueState, discover_queue
from qa.runtime_matrix.common import write_row


def _partial_write_case(case_dir: Path) -> dict:
    input_dir = case_dir / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)
    source = input_dir / "partial.wav"
    partial_bytes = b"partial"
    complete_bytes = b"complete media bytes"
    source.write_bytes(partial_bytes)
    state_path = case_dir / "queue_state.json"
    state = QueueState(state_path)
    wait_calls: list[dict[str, str | float]] = []

    original_wait = queue_manager.wait_for_stable_file
    try:
        def finish_write(path: Path, seconds: float) -> None:
            wait_calls.append({"path": str(path), "seconds": seconds})
            source.write_bytes(complete_bytes)

        queue_manager.wait_for_stable_file = finish_write
        queued = discover_queue([input_dir], {".wav"}, recursive=True, stability_seconds=1.0, state=state, skip_done=True)
    finally:
        queue_manager.wait_for_stable_file = original_wait

    item = QueueState(state_path).data["items"][0]
    failures: list[str] = []
    if queued != [source]:
        failures.append(f"partial-write case queued unexpected paths: {[str(path) for path in queued]}")
    if not wait_calls:
        failures.append("partial-write case did not wait for file stability")
    if f"|{len(complete_bytes)}|" not in item.get("fast_key", ""):
        failures.append("partial-write case recorded fast key before final bytes were visible")
    return {
        "source": str(source),
        "state_path": str(state_path),
        "queued": [str(path) for path in queued],
        "wait_calls": wait_calls,
        "partial_bytes": len(partial_bytes),
        "complete_bytes": len(complete_bytes),
        "recorded_fast_key": item.get("fast_key", ""),
        "failures": failures,
    }


def _repeat_poll_case(case_dir: Path) -> dict:
    input_dir = case_dir / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)
    source = input_dir / "drop.wav"
    source.write_bytes(b"stable media")
    state_path = case_dir / "queue_state.json"
    state = QueueState(state_path)

    original_wait = queue_manager.wait_for_stable_file
    try:
        queue_manager.wait_for_stable_file = lambda _path, _seconds: None
        first = discover_queue([input_dir], {".wav"}, recursive=True, stability_seconds=0, state=state, skip_done=False)
        second = discover_queue([input_dir], {".wav"}, recursive=True, stability_seconds=0, state=state, skip_done=False)
    finally:
        queue_manager.wait_for_stable_file = original_wait

    items = QueueState(state_path).data["items"]
    source_count = sum(1 for item in items if item.get("source_path") == str(source.resolve()))
    failures: list[str] = []
    if first != [source]:
        failures.append(f"repeat-poll case first poll queued unexpected paths: {[str(path) for path in first]}")
    if second != [source]:
        failures.append(f"repeat-poll case second poll queued unexpected paths: {[str(path) for path in second]}")
    if len(items) != 1 or source_count != 1:
        failures.append("repeat-poll case duplicated queue state for the same source path")
    return {
        "source": str(source),
        "state_path": str(state_path),
        "first_poll": [str(path) for path in first],
        "second_poll": [str(path) for path in second],
        "queue_item_count": len(items),
        "source_count": source_count,
        "failures": failures,
    }


def _done_fast_key_case(case_dir: Path) -> dict:
    input_dir = case_dir / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)
    source = input_dir / "done.wav"
    source.write_bytes(b"already processed")
    state_path = case_dir / "queue_state.json"
    state = QueueState(state_path)
    fast_key = queue_manager.file_key(source)
    state.upsert(QueueItem(str(source.resolve()), "already-done", fast_key, status="done"))
    hash_called = False

    original_wait = queue_manager.wait_for_stable_file
    original_hash = queue_manager.sha256_file
    try:
        queue_manager.wait_for_stable_file = lambda _path, _seconds: None

        def fail_hash(path: Path) -> str:
            nonlocal hash_called
            hash_called = True
            raise AssertionError(f"completed fast key should not be hashed: {path}")

        queue_manager.sha256_file = fail_hash
        queued = discover_queue([input_dir], {".wav"}, recursive=True, stability_seconds=0, state=state, skip_done=True)
    finally:
        queue_manager.wait_for_stable_file = original_wait
        queue_manager.sha256_file = original_hash

    failures: list[str] = []
    if queued:
        failures.append(f"done-fast-key case queued completed paths: {[str(path) for path in queued]}")
    if hash_called:
        failures.append("done-fast-key case rehashed a completed fast key")
    return {
        "source": str(source),
        "state_path": str(state_path),
        "queued": [str(path) for path in queued],
        "done_fast_key": fast_key,
        "sha256_file_called": hash_called,
        "failures": failures,
    }


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    partial = _partial_write_case(evidence_dir / "partial_write")
    repeat = _repeat_poll_case(evidence_dir / "repeat_poll")
    done_fast = _done_fast_key_case(evidence_dir / "done_fast_key")
    failures = partial["failures"] + repeat["failures"] + done_fast["failures"]

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "Watched-folder queue waits through partial writes, avoids duplicate queue state on repeat polls, and skips completed fast keys without hashing."
            if not failures
            else "Watched-folder queue race contract failed."
        ),
        details={
            "partial_write": partial,
            "repeat_poll": repeat,
            "done_fast_key_skip": done_fast,
            "failures": failures,
        },
        artifacts=[
            Path(partial["state_path"]),
            Path(repeat["state_path"]),
            Path(done_fast["state_path"]),
        ],
    )
