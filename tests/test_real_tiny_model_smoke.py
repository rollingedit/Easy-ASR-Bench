import json
from pathlib import Path

import pytest

from qa.run_real_tiny_model_smoke import REFERENCE_TEXT, assert_smoke_report


def write_smoke_report(tmp_path: Path, transcript: str) -> Path:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    for name in ["results.txt", "benchmark.csv", "compare.html"]:
        (report_dir / name).write_text("fixture", encoding="utf-8")
    (report_dir / "results.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "model": {"candidate_id": "tiny", "display_name": "tiny"},
                        "transcript_chunks": [{"chunk_id": "0001", "text": transcript}],
                        "metrics": {"vram_measurement_source": "unavailable"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return report_dir


def test_real_tiny_smoke_requires_nonempty_transcript_and_wer_threshold(tmp_path: Path):
    report_dir = write_smoke_report(tmp_path, REFERENCE_TEXT)

    results = assert_smoke_report(report_dir, REFERENCE_TEXT, max_normalized_wer=0.0)

    metrics = results["runs"][0]["metrics"]
    assert metrics["real_smoke_reference_text"] == REFERENCE_TEXT
    assert metrics["real_smoke_normalized_wer"] == 0.0
    assert metrics["real_smoke_max_normalized_wer"] == 0.0


def test_real_tiny_smoke_fails_when_transcript_accuracy_is_too_low(tmp_path: Path):
    report_dir = write_smoke_report(tmp_path, "totally unrelated words")

    with pytest.raises(AssertionError, match="normalized WER"):
        assert_smoke_report(report_dir, REFERENCE_TEXT, max_normalized_wer=0.25)
