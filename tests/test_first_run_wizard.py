from pathlib import Path

from app.adapters.base import ModelCandidate
from app.first_run import build_first_run_smoke_report, run_first_run_wizard


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


def test_first_run_initial_paste_hf_action_skips_dead_end_menu(monkeypatch, tmp_path: Path):
    calls = []
    config = {"folders": {"models": str(tmp_path / "Models"), "input": str(tmp_path / "Input")}}

    monkeypatch.setattr("app.first_run.scan_models", lambda root: ([], []))
    monkeypatch.setattr("app.first_run.choose_one", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("menu should not be shown")))
    monkeypatch.setattr("app.first_run.download_hf_model_interactive", lambda root, input_func=input, print_func=print: calls.append(root) or (root / "hf"))

    assert run_first_run_wizard(config, input_func=lambda prompt: "unused", initial_action="paste_hf") is True
    assert calls == [tmp_path / "Models"]


def test_first_run_baseline_discloses_dependency_group(monkeypatch, tmp_path: Path, capsys):
    config = {"folders": {"models": str(tmp_path / "Models"), "input": str(tmp_path / "Input")}}

    monkeypatch.setattr("app.first_run.scan_models", lambda root: ([], []))
    monkeypatch.setattr("app.first_run.choose_one", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.first_run.download_recommended_baseline", lambda root, input_func=input, print_func=print: root / "baseline")

    assert run_first_run_wizard(config, input_func=lambda prompt: "d") is True

    output = capsys.readouterr().out
    assert "faster-whisper / CTranslate2 runtime" in output
    assert "Runs on CPU by default" in output


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


def test_first_run_smoke_report_is_noninteractive_and_actionable(monkeypatch, tmp_path: Path):
    config = {"folders": {"models": str(tmp_path / "Models"), "input": str(tmp_path / "Input")}}

    monkeypatch.setattr("app.first_run.scan_models", lambda root: ([], []))

    report = build_first_run_smoke_report(config)

    assert report["schema"] == "easy_asr_bench.first_run_smoke.v1"
    assert report["network_used"] is False
    assert report["dead_end"] is False
    assert report["recommended_next_action"] == "download_recommended_baseline"
    assert "paste_hugging_face_link" in report["available_actions"]
