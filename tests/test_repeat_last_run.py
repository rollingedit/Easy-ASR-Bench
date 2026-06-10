from argparse import Namespace
from pathlib import Path

from app.adapters.base import ModelCandidate
from app.model_selector import LAST_RUN_SELECTION_SCHEMA


def candidate(candidate_id: str, tmp_path: Path) -> ModelCandidate:
    return ModelCandidate(
        candidate_id=candidate_id,
        display_name=candidate_id,
        family_name="Test",
        backend="fixture",
        container_format="fixture",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="32-bit / FP32",
        path=tmp_path / candidate_id,
        adapter_name="fixture_asr",
        runnable=True,
        category="asr",
    )


def args_for(tmp_path: Path) -> Namespace:
    return Namespace(
        paths=[str(tmp_path / "audio.wav")],
        config=str(tmp_path / "config.json"),
        interactive=False,
        scan_only=False,
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


def test_noninteractive_paths_use_saved_last_run_selection(monkeypatch, tmp_path: Path):
    from app import main

    first = candidate("first", tmp_path)
    second = candidate("second", tmp_path)
    config = {
        "folders": {"models": str(tmp_path / "Models"), "input": str(tmp_path / "Input"), "output": str(tmp_path / "Output"), "temp": str(tmp_path / "Temp"), "logs": str(tmp_path / "Logs"), "cache": str(tmp_path / "Cache")},
        "input": {"extensions": [".wav"], "recursive_folders": True, "file_stability_wait_seconds": 0},
        "runtime": {},
        "advanced": {"keep_temp_wavs": False},
        "last_run_selection": {"schema": LAST_RUN_SELECTION_SCHEMA, "candidate_ids": ["second"], "reference_llm_candidate_id": ""},
    }
    seen = {}

    monkeypatch.setattr("app.main.load_config", lambda path: config)
    monkeypatch.setattr("app.main.check_for_updates_from_config", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr("app.main.scan_models", lambda root: ([first, second], []))
    monkeypatch.setattr("app.main.setup_logging", lambda path: None)
    monkeypatch.setattr("app.main.warn_runtime_dependency_fallbacks", lambda config: None)
    monkeypatch.setattr("app.main.ensure_dependencies", lambda selected, config, reference_llm=None: (seen.setdefault("ids", [item.candidate_id for item in selected]) and selected, reference_llm))
    monkeypatch.setattr("app.main.collect_input_files", lambda args, config: [])

    main._main(args_for(tmp_path))

    assert seen["ids"] == ["second"]


def test_noninteractive_paths_stop_on_stale_saved_last_run(monkeypatch, tmp_path: Path, capsys):
    from app import main

    config = {
        "folders": {"models": str(tmp_path / "Models"), "input": str(tmp_path / "Input"), "output": str(tmp_path / "Output"), "temp": str(tmp_path / "Temp"), "logs": str(tmp_path / "Logs"), "cache": str(tmp_path / "Cache")},
        "input": {"extensions": [".wav"], "recursive_folders": True, "file_stability_wait_seconds": 0},
        "runtime": {},
        "advanced": {"keep_temp_wavs": False},
        "last_run_selection": {"schema": LAST_RUN_SELECTION_SCHEMA, "candidate_ids": ["missing"], "reference_llm_candidate_id": ""},
    }
    called = {"ensure": False}

    monkeypatch.setattr("app.main.load_config", lambda path: config)
    monkeypatch.setattr("app.main.check_for_updates_from_config", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr("app.main.scan_models", lambda root: ([candidate("available", tmp_path)], []))
    monkeypatch.setattr("app.main.setup_logging", lambda path: None)
    monkeypatch.setattr("app.main.warn_runtime_dependency_fallbacks", lambda config: None)
    monkeypatch.setattr("app.main.ensure_dependencies", lambda *args, **kwargs: called.__setitem__("ensure", True))

    main._main(args_for(tmp_path))
    output = capsys.readouterr().out

    assert called["ensure"] is False
    assert "Saved last-run model selection is stale" in output
