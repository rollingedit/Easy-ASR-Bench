import argparse
from pathlib import Path

import pytest

from app import doctor, main as app_main


def test_app_main_doctor_forwards_strict_and_json(monkeypatch):
    captured = {}

    def fake_load_config(path: Path):
        return {}

    def fake_run_doctor(config_path: Path, **kwargs):
        captured["config_path"] = config_path
        captured.update(kwargs)
        return 7

    monkeypatch.setattr(app_main, "load_config", fake_load_config)
    monkeypatch.setattr(doctor, "run_doctor", fake_run_doctor)

    args = argparse.Namespace(
        config="config.json",
        doctor=True,
        strict=True,
        json=True,
        repair_plan=False,
        repair_all_safe=False,
        validate_real_smoke=False,
        install_deps=False,
        allow_downloads=False,
        no_network=False,
        full_real_smoke=False,
    )

    with pytest.raises(SystemExit) as exc:
        app_main._main(args)

    assert exc.value.code == 7
    assert captured["config_path"] == Path("config.json")
    assert captured["strict"] is True
    assert captured["json_output"] is True


def test_download_model_flag_rescans_without_requiring_restart(monkeypatch, tmp_path: Path):
    calls = {"scan": 0, "summary": 0}
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"folders":{"models":"Models","input":"Input","output":"Output","temp":"Temp","logs":"Logs","cache":"Cache"}}\n',
        encoding="utf-8",
    )

    def fake_download(models_root: Path):
        assert models_root == Path("Models")
        return models_root / "downloaded-model"

    def fake_scan(models_root: Path):
        calls["scan"] += 1
        assert models_root == Path("Models")
        return [], []

    def fake_summary(runnable, unsupported):
        calls["summary"] += 1
        assert runnable == []
        assert unsupported == []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_main, "download_hf_model_interactive", fake_download)
    monkeypatch.setattr(app_main, "scan_models", fake_scan)
    monkeypatch.setattr(app_main, "print_scan_summary", fake_summary)

    args = argparse.Namespace(
        paths=[],
        config=str(config_path),
        interactive=False,
        scan_only=True,
        doctor=False,
        json=False,
        strict=False,
        repair_plan=False,
        repair_all_safe=False,
        validate_real_smoke=False,
        install_deps=False,
        allow_downloads=False,
        no_network=False,
        full_real_smoke=False,
        first_run=False,
        first_run_smoke=False,
        download_model=True,
        download_model_first=False,
        open_models=False,
        open_input=False,
        watch=False,
        once=False,
    )

    app_main._main(args)

    assert args.interactive is True
    assert calls == {"scan": 1, "summary": 1}
