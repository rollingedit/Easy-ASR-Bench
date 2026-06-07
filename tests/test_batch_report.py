import json
from pathlib import Path

from app.batch_report import render_batch_html, write_batch_report


def make_result(folder: Path, model_name: str, model_count: int = 1) -> None:
    folder.mkdir(parents=True)
    runs = []
    for index in range(model_count):
        runs.append(
            {
                "model": {"display_name": f"{model_name} {index + 1}", "backend": "fixture", "precision": "fp32"},
                "metrics": {"audio_seconds_per_wall_second": 2.5, "peak_process_memory_mb": 120, "peak_vram_mb": 400},
                "errors": [],
            }
        )
    (folder / "results.json").write_text(
        json.dumps(
            {
                "source": {"duration_seconds": 12.5},
                "chunk_plan": {"chunks": [{"chunk_id": "0001"}, {"chunk_id": "0002"}]},
                "runs": runs,
                "unsupported_models": [{"display_name": "Skipped"}],
            }
        ),
        encoding="utf-8",
    )
    (folder / "compare.html").write_text("<!doctype html>", encoding="utf-8")


def test_batch_report_renders_side_by_side_file_cards_with_model_summaries(tmp_path: Path):
    first = tmp_path / "out1"
    second = tmp_path / "out2"
    make_result(first, "Model One")
    make_result(second, "Model Two")
    payload = {
        "files": [
            {"source_path": str(tmp_path / "a.wav"), "status": "done", "output_path": str(first)},
            {"source_path": str(tmp_path / "b.mp4"), "status": "done", "output_path": str(second)},
        ]
    }

    html = render_batch_html(payload, tmp_path)

    assert "file-grid" in html
    assert "file-card" in html
    assert "Model One 1" in html
    assert "Model Two 1" in html
    assert "xRT" in html
    assert "Open report" in html


def test_write_batch_report_writes_index_and_json(tmp_path: Path):
    report_dir = write_batch_report(tmp_path, [{"source_path": "a.wav", "status": "failed", "output_path": ""}])

    assert (report_dir / "index.html").exists()
    assert json.loads((report_dir / "batch.json").read_text(encoding="utf-8"))["files"][0]["status"] == "failed"


def test_batch_report_json_scripts_are_parseable_json_not_html_entities(tmp_path: Path):
    payload = {"schema": "easy_asr_bench.batch_report.v1", "files": [{"source_path": "a&b.wav", "status": "done"}]}

    html = render_batch_html(payload, tmp_path)

    assert "&quot;" not in html
    assert '<script type="application/json" id="batch-json">{"schema"' in html
    assert "\\u0026" in html


def test_batch_report_pages_many_files_and_scrolls_many_models(tmp_path: Path):
    files = []
    for index in range(20):
        folder = tmp_path / f"out{index}"
        make_result(folder, f"Model {index}", model_count=12)
        files.append({"source_path": str(tmp_path / f"file{index}.wav"), "status": "done", "output_path": str(folder)})

    html = render_batch_html({"files": files}, tmp_path)

    assert "const batchPageSize = 6" in html
    assert "max-height:260px" in html
    assert "Filter by file path or status" in html
    assert "Files ${filtered.length ? batchPage * batchPageSize + 1 : 0}-${Math.min(filtered.length, (batchPage + 1) * batchPageSize)} of ${filtered.length}" in html
