import json

from app.main import output_status


def test_output_status_marks_failed_file_report_failed(tmp_path):
    report = tmp_path / "report"
    report.mkdir()
    (report / "results.json").write_text(
        json.dumps({"runs": [], "errors": [{"status": "failed_before_model_run"}]}),
        encoding="utf-8",
    )

    assert output_status(report) == "failed"


def test_output_status_marks_model_error_report_done(tmp_path):
    report = tmp_path / "report"
    report.mkdir()
    (report / "results.json").write_text(
        json.dumps({"runs": [{"errors": ["model failed"]}], "errors": []}),
        encoding="utf-8",
    )

    assert output_status(report) == "done"
