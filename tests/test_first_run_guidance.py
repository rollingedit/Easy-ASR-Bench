from pathlib import Path

from app.adapters.base import ModelCandidate
from app.main import print_first_run_guidance


def test_first_run_guidance_for_empty_models_points_to_next_steps(tmp_path, capsys):
    config = {"folders": {"input": str(tmp_path / "Input")}}

    print_first_run_guidance(config, [], [], tmp_path / "Models")

    output = capsys.readouterr().out
    assert "No runnable ASR model is installed yet." in output
    assert "[P]" in output
    assert "[M]" in output
    assert "[I]" in output
    assert "[Q]" in output
    assert "Paste a Hugging Face model/repo link" in output
    assert "Open" in output and "Models" in output
    assert "Unsupported or incomplete files" in output


def test_first_run_guidance_with_runnable_model_points_to_input(tmp_path, capsys):
    model = ModelCandidate(
        candidate_id="fixture",
        display_name="Fixture",
        family_name="Fixture",
        backend="fixture",
        container_format="fixture",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="32-bit / FP32",
        path=Path("fixture"),
        adapter_name="fixture",
        runnable=True,
    )
    config = {"folders": {"input": str(tmp_path / "Input")}}

    print_first_run_guidance(config, [model], [], tmp_path / "Models")

    output = capsys.readouterr().out
    assert "Runnable ASR model found." in output
    assert "Put audio/video files" in output
