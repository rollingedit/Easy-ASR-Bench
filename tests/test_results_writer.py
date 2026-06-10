import json
from pathlib import Path

import pytest

from app import results_writer


def minimal_results(name: str = "sample.wav", sha256: str = "abc") -> dict:
    return {
        "schema": "easy_asr_bench.results.v1",
        "app_version": "0.0.test",
        "created_local": "2026-06-10 00:00:00",
        "source": {
            "path": str(Path("Input") / name),
            "name": name,
            "sha256": sha256,
            "duration_seconds": 1.0,
        },
        "environment": {},
        "dependency_versions": {},
        "adapter_versions": {},
        "chunk_plan": {
            "sample_rate": 16000,
            "source_audio_seconds": 1.0,
            "chunks": [{"chunk_id": "0001", "index": 0, "start_seconds": 0.0, "end_seconds": 1.0}],
        },
        "selected_models": [],
        "runs": [],
        "unsupported_models": [],
        "pairwise_differences": {},
        "runtime_rankings": {"schema": "easy_asr_bench.runtime_rankings.v1", "rows": []},
        "errors": [],
    }


def test_write_all_reports_uses_collision_proof_report_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(results_writer, "_report_timestamp", lambda: "20260610_010203_000004")

    first = results_writer.write_all_reports(minimal_results(), tmp_path)
    second = results_writer.write_all_reports(minimal_results(), tmp_path)

    assert first != second
    assert first.name.startswith("sample__20260610_010203_000004__")
    assert second.name.endswith("__01")
    assert (first / "results.json").exists()
    assert (second / "results.json").exists()


def test_write_all_reports_uses_source_identity_hash_for_same_stem(tmp_path, monkeypatch):
    monkeypatch.setattr(results_writer, "_report_timestamp", lambda: "20260610_010203_000004")

    first = results_writer.write_all_reports(minimal_results("same.wav", "aaa"), tmp_path)
    second = results_writer.write_all_reports(minimal_results("same.wav", "bbb"), tmp_path)

    assert first != second
    assert not second.name.endswith("__01")


def test_write_all_reports_cleans_partial_directory_on_failure(tmp_path, monkeypatch):
    def fail_render(results):
        raise RuntimeError("render failed")

    monkeypatch.setattr(results_writer, "render_text_report", fail_render)

    with pytest.raises(RuntimeError, match="render failed"):
        results_writer.write_all_reports(minimal_results(), tmp_path)

    assert not [path for path in tmp_path.iterdir() if path.is_dir()]


def test_write_all_reports_publishes_only_after_required_files_exist(tmp_path):
    output = results_writer.write_all_reports(minimal_results(), tmp_path)

    assert set(results_writer.REQUIRED_REPORT_FILES) <= {path.name for path in output.iterdir() if path.is_file()}
    assert not list(tmp_path.glob(".*.partial"))
    assert json.loads((output / "results.json").read_text(encoding="utf-8"))["source"]["name"] == "sample.wav"
