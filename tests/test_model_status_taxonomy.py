from pathlib import Path

from app.adapters.base import ModelCandidate
from app.main import print_scan_summary
from app.model_status import model_status, model_status_label


def candidate(**overrides) -> ModelCandidate:
    values = {
        "candidate_id": "model",
        "display_name": "Model",
        "family_name": "Model",
        "backend": "test",
        "container_format": "test",
        "task": "automatic-speech-recognition",
        "precision": "fp32",
        "quantization_label": "FP32",
        "path": Path("Models/Model"),
        "adapter_name": "none",
        "runnable": False,
    }
    values.update(overrides)
    return ModelCandidate(**values)


def test_model_status_maps_probe_dependency_reference_and_unsafe_states():
    assert model_status(candidate(runnable=True, category="asr")) == "runnable_asr"
    assert model_status(candidate(runnable_after_dependency_install=True)) == "needs_dependency_install"
    assert model_status(candidate(category="asr_probe_required")) == "asr_probe_required"
    assert model_status(candidate(category="reference_llm")) == "reference_llm"
    assert model_status(candidate(category="unsupported_llm")) == "unsupported_llm_format"
    assert model_status(candidate(path=Path("Models/model.pt"), warnings=["pickle checkpoint blocked"])) == "unsafe_blocked"


def test_scan_summary_prints_explicit_user_facing_buckets(capsys):
    runnable = [candidate(runnable=True, category="asr", display_name="Runnable")]
    unsupported = [
        candidate(display_name="Probe", category="asr_probe_required"),
        candidate(display_name="LLM", category="unsupported_llm"),
        candidate(display_name="Missing", missing_files=["config.json"]),
    ]

    print_scan_summary(runnable, unsupported)

    output = capsys.readouterr().out
    assert "Runnable ASR candidates:" in output
    assert "ASR probe required:" in output
    assert "Unsupported LLM format" in output
    assert "Recognized incomplete:" in output
    assert "Missing: config.json" in output
    assert model_status_label(unsupported[0]) == "ASR probe required"
