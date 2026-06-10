import argparse
from pathlib import Path

import pytest

from app import doctor, main as app_main


def main_args(config: str = "config.json") -> argparse.Namespace:
    return argparse.Namespace(
        paths=[],
        config=config,
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
        download_model=False,
        download_model_first=False,
        open_models=False,
        open_input=False,
        watch=False,
        once=False,
    )


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


def test_app_main_runs_update_check_from_config(monkeypatch, tmp_path: Path):
    calls = {"update": 0}
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"app":{"check_for_updates_on_run":true},"folders":{"models":"Models","input":"Input","output":"Output","temp":"Temp","logs":"Logs","cache":"Cache"}}\n',
        encoding="utf-8",
    )

    def fake_update(config, *, context, print_func=print):
        calls["update"] += 1
        assert context == "run"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("app.update_check.check_for_updates_from_config", fake_update)
    monkeypatch.setattr(app_main, "scan_models", lambda models_root: ([], []))
    monkeypatch.setattr(app_main, "print_scan_summary", lambda runnable, unsupported: None)

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
        download_model=False,
        download_model_first=False,
        open_models=False,
        open_input=False,
        watch=False,
        once=False,
    )

    app_main._main(args)

    assert calls["update"] == 1


def test_main_crash_handler_logs_config_load_failure(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: main_args("missing-config.json"))
    monkeypatch.setattr(app_main, "load_config", lambda path: (_ for _ in ()).throw(RuntimeError("config exploded")))

    app_main.main()

    output = capsys.readouterr().out
    crash_logs = list((tmp_path / "Logs").glob("crash_*.log"))
    assert crash_logs
    assert "unexpected error" in output
    assert "setup.bat --doctor --strict" in output
    assert "issues/new/choose" in output
    assert "config exploded" in crash_logs[0].read_text(encoding="utf-8")


def test_main_crash_handler_logs_model_scan_failure(monkeypatch, tmp_path: Path, capsys):
    config_path = tmp_path / "config.json"
    logs = tmp_path / "CustomLogs"
    config_path.write_text(
        '{"folders":{"models":"Models","input":"Input","output":"Output","temp":"Temp","logs":"' + str(logs).replace("\\", "\\\\") + '","cache":"Cache"}}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: main_args(str(config_path)))
    monkeypatch.setattr("app.update_check.check_for_updates_from_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_main, "scan_models", lambda root: (_ for _ in ()).throw(RuntimeError("scan exploded")))

    app_main.main()

    assert "scan exploded" in capsys.readouterr().out
    crash_logs = list(logs.glob("crash_*.log"))
    assert crash_logs
    assert "scan exploded" in crash_logs[0].read_text(encoding="utf-8")


def test_main_crash_handler_logs_report_write_failure(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: main_args())
    monkeypatch.setattr(app_main, "_main", lambda args: (_ for _ in ()).throw(RuntimeError("report write exploded")))

    app_main.main()

    output = capsys.readouterr().out
    crash_logs = list((tmp_path / "Logs").glob("crash_*.log"))
    assert crash_logs
    assert "report write exploded" in output
    assert "report write exploded" in crash_logs[0].read_text(encoding="utf-8")
