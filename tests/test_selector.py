from pathlib import Path

from app.adapters.base import ModelCandidate
from app.model_selector import choose_candidates, choose_probe_candidates, parse_selection


def test_parse_selection_range():
    assert parse_selection("1-3", 5) == [1, 2, 3]


def test_parse_selection_compact_single_digits():
    assert parse_selection("123", 5) == [1, 2, 3]


def test_choose_candidates_download_path_rescans(monkeypatch, tmp_path: Path):
    downloaded = {"done": False}

    def fake_download(models_root: Path):
        downloaded["done"] = True
        return models_root / "downloaded"

    def fake_scan(models_root: Path):
        return [
            ModelCandidate(
                candidate_id="downloaded",
                display_name="Downloaded",
                family_name="Downloaded",
                backend="transformers",
                container_format="safetensors",
                task="automatic-speech-recognition",
                precision="fp32",
                quantization_label="32-bit / FP32",
                path=models_root / "downloaded",
                adapter_name="hf_transformers_asr",
                runnable=True,
            )
        ], []

    answers = iter(["d", "1", "a", "4"])
    monkeypatch.setattr("app.model_selector.download_hf_model_interactive", fake_download)
    monkeypatch.setattr("app.model_selector.scan_models", fake_scan)
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    selected, reference_llm = choose_candidates([], [], {"llm_reference": {"custom_model_paths": []}}, tmp_path / "config.json", tmp_path)

    assert downloaded["done"] is True
    assert [candidate.candidate_id for candidate in selected] == ["downloaded"]
    assert reference_llm is None


def test_choose_probe_candidates_selects_probe_required_folder(monkeypatch, tmp_path: Path):
    probe = ModelCandidate(
        candidate_id="probe",
        display_name="Probe",
        family_name="Probe",
        backend="transformers",
        container_format="safetensors",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="32-bit / FP32",
        path=tmp_path / "probe",
        adapter_name="hf_transformers_asr",
        runnable=False,
        category="asr_probe_required",
    )
    monkeypatch.setattr("builtins.input", lambda _: "1")

    selected = choose_probe_candidates([probe])

    assert selected == [probe]
