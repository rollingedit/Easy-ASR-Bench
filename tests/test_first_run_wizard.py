from pathlib import Path

from app.adapters.base import ModelCandidate
from app.first_run import run_first_run_wizard


def test_first_run_empty_models_downloads_recommended_baseline(monkeypatch, tmp_path: Path, capsys):
    calls = []
    config = {"folders": {"models": str(tmp_path / "Models"), "input": str(tmp_path / "Input")}}

    monkeypatch.setattr("app.first_run.scan_models", lambda root: ([], []))
    monkeypatch.setattr("app.first_run.choose_one", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.first_run.download_recommended_baseline", lambda root, input_func=input, print_func=print: calls.append(root) or (root / "baseline"))

    should_continue = run_first_run_wizard(config, input_func=lambda prompt: "d")

    assert should_continue is True
    assert calls == [tmp_path / "Models"]
    output = capsys.readouterr().out
    assert "No runnable ASR model is installed yet." in output
    assert "Download recommended CPU baseline" in output


def test_first_run_empty_models_can_paste_hf_link(monkeypatch, tmp_path: Path):
    calls = []
    config = {"folders": {"models": str(tmp_path / "Models"), "input": str(tmp_path / "Input")}}

    monkeypatch.setattr("app.first_run.scan_models", lambda root: ([], []))
    monkeypatch.setattr("app.first_run.choose_one", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.first_run.download_hf_model_interactive", lambda root, input_func=input, print_func=print: calls.append(root) or (root / "hf"))

    assert run_first_run_wizard(config, input_func=lambda prompt: "p") is True
    assert calls == [tmp_path / "Models"]


def test_first_run_existing_model_runs_now(monkeypatch, tmp_path: Path):
    model = ModelCandidate(
        candidate_id="asr",
        display_name="ASR",
        family_name="ASR",
        backend="fixture",
        container_format="fixture",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="32-bit / FP32",
        path=tmp_path / "model",
        adapter_name="fixture",
        runnable=True,
    )
    config = {"folders": {"models": str(tmp_path / "Models"), "input": str(tmp_path / "Input")}}

    monkeypatch.setattr("app.first_run.scan_models", lambda root: ([model], []))
    monkeypatch.setattr("app.first_run.choose_one", lambda *args, **kwargs: None)

    assert run_first_run_wizard(config, input_func=lambda prompt: "r") is True
