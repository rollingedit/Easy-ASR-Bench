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
