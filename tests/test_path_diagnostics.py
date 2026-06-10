import json
from pathlib import Path

from app.doctor import build_doctor_report, run_doctor
from app.path_diagnostics import build_path_diagnostics


def test_path_diagnostics_warn_on_non_ascii_profile_and_onedrive_install(tmp_path: Path):
    report = build_path_diagnostics(
        {"folders": {"models": "Models"}},
        project_root=tmp_path / "OneDrive" / "Easy ASR Bench",
        env={"USERPROFILE": str(tmp_path / "Users" / "Jose\u0301"), "OneDrive": str(tmp_path / "OneDrive")},
    )

    codes = {warning["code"] for warning in report["warnings"]}
    assert "non_ascii_path" in codes
    assert "onedrive_or_redirected_path" in codes
    assert report["ok"] is False


def test_path_diagnostics_clean_ascii_paths_have_no_warnings(tmp_path: Path):
    report = build_path_diagnostics(
        {"folders": {"models": "Models", "temp": "Temp"}},
        project_root=tmp_path / "Easy-ASR-Bench",
        env={"USERPROFILE": str(tmp_path / "Users" / "PublicUser")},
    )

    assert report["warnings"] == []
    assert report["ok"] is True


def test_doctor_json_includes_path_diagnostics(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "app": {"version": "0.4.0", "version_channel": "prerelease"},
                "folders": {"models": "Models", "input": "Input", "output": "Output", "temp": "Temp", "logs": "Logs", "cache": "Cache"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    report = build_doctor_report(config_path)

    assert report["path_diagnostics"]["schema"] == "easy_asr_bench.path_diagnostics.v1"
    assert any(record["kind"] == "install_root" for record in report["path_diagnostics"]["records"])


def test_doctor_text_prints_path_warnings(tmp_path: Path, monkeypatch, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "app": {"version": "0.4.0", "version_channel": "prerelease"},
                "folders": {"models": "Models", "input": "Input", "output": "Output", "temp": "Temp", "logs": "Logs", "cache": "Cache"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "app.doctor.build_path_diagnostics",
        lambda config, project_root: {
            "schema": "easy_asr_bench.path_diagnostics.v1",
            "records": [],
            "warnings": [
                {
                    "kind": "install_root",
                    "path": str(tmp_path),
                    "code": "onedrive_or_redirected_path",
                    "message": "Path appears to be under OneDrive or a redirected profile.",
                }
            ],
            "ok": False,
        },
    )

    exit_code = run_doctor(config_path)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Path warnings:" in output
    assert "OneDrive" in output
