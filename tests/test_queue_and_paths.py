from pathlib import Path

from app.queue_manager import QueueItem, QueueState, discover_queue
from app.utils import parse_windows_path_list


def test_parse_windows_paths_with_spaces_unicode_and_apostrophe():
    raw = '"C:\\Users\\Me\\Long File.wav" "D:\\音声\\clip one.mp3" "E:\\it\\Bob\'s sample.mp4"'

    paths = parse_windows_path_list(raw)

    assert [str(path) for path in paths] == [
        "C:\\Users\\Me\\Long File.wav",
        "D:\\音声\\clip one.mp3",
        "E:\\it\\Bob's sample.mp4",
    ]


def test_discover_queue_skips_done_fast_key_without_rehashing(tmp_path: Path, monkeypatch):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"same")
    state = QueueState(tmp_path / "state.json")
    fast_key = f"{source.resolve()}|{source.stat().st_size}|{int(source.stat().st_mtime)}"
    state.upsert(QueueItem(str(source.resolve()), "already-done", fast_key, status="done"))

    called = {"hash": False}

    def fail_hash(path):
        called["hash"] = True
        raise AssertionError("sha256_file should not be called for a completed fast key")

    monkeypatch.setattr("app.queue_manager.wait_for_stable_file", lambda path, seconds: None)
    monkeypatch.setattr("app.queue_manager.sha256_file", fail_hash)

    assert discover_queue([tmp_path], {".wav"}, recursive=True, stability_seconds=0, state=state, skip_done=True) == []
    assert called["hash"] is False


def test_discover_queue_queues_new_file_by_fast_key_before_full_hash(tmp_path: Path, monkeypatch):
    source = tmp_path / "large.wav"
    source.write_bytes(b"media")
    state = QueueState(tmp_path / "state.json")

    called = {"hash": False}

    def fail_hash(path):
        called["hash"] = True
        raise AssertionError("new files should not be fully hashed before processing starts")

    monkeypatch.setattr("app.queue_manager.wait_for_stable_file", lambda path, seconds: None)
    monkeypatch.setattr("app.queue_manager.sha256_file", fail_hash)

    queued = discover_queue([tmp_path], {".wav"}, recursive=True, stability_seconds=0, state=state, skip_done=True)

    assert queued == [source]
    assert called["hash"] is False
    item = QueueState(tmp_path / "state.json").data["items"][0]
    assert item["sha256"] == ""
    assert item["fast_key"]


def test_queue_state_preserves_failed_item_for_resume_visibility(tmp_path: Path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"bad")
    state = QueueState(tmp_path / "state.json")
    state.upsert(QueueItem(str(source.resolve()), "hash", "fast"))
    state.mark(source.resolve(), "failed", error="model failed")

    reloaded = QueueState(tmp_path / "state.json")
    item = reloaded.data["items"][0]
    assert item["status"] == "failed"
    assert item["errors"] == ["model failed"]
