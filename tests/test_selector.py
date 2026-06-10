from pathlib import Path

from app.adapters.base import ModelCandidate
from app.model_selector import LAST_RUN_SELECTION_SCHEMA, choose_candidates, choose_probe_candidates, parse_selection, resolve_last_run_selection


def candidate(candidate_id: str, tmp_path: Path, *, reference: bool = False) -> ModelCandidate:
    return ModelCandidate(
        candidate_id=candidate_id,
        display_name=candidate_id,
        family_name="Test",
        backend="fixture",
        container_format="fixture",
        task="text-generation" if reference else "automatic-speech-recognition",
        precision="fp32",
        quantization_label="32-bit / FP32",
        path=tmp_path / candidate_id,
        adapter_name="gguf_llm_reference" if reference else "fixture_asr",
        runnable=not reference,
        category="reference_llm" if reference else "asr",
    )


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


def test_choose_candidates_saves_last_run_selection(monkeypatch, tmp_path: Path):
    asr = candidate("asr_one", tmp_path)
    config = {"llm_reference": {"custom_model_paths": []}}
    answers = iter(["1", "a", "4"])
    config_path = tmp_path / "config.json"

    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    selected, reference_llm = choose_candidates([asr], [], config, config_path, tmp_path)

    assert [item.candidate_id for item in selected] == ["asr_one"]
    assert reference_llm is None
    assert config["last_run_selection"]["schema"] == LAST_RUN_SELECTION_SCHEMA
    assert config["last_run_selection"]["candidate_ids"] == ["asr_one"]
    assert "asr_one" in config_path.read_text(encoding="utf-8")


def test_choose_candidates_blank_enter_reuses_valid_last_run(monkeypatch, tmp_path: Path):
    first = candidate("first", tmp_path)
    second = candidate("second", tmp_path)
    config = {
        "llm_reference": {"custom_model_paths": []},
        "last_run_selection": {"schema": LAST_RUN_SELECTION_SCHEMA, "candidate_ids": ["second"], "reference_llm_candidate_id": ""},
    }
    monkeypatch.setattr("builtins.input", lambda _: "")

    selected, reference_llm = choose_candidates([first, second], [], config, tmp_path / "config.json", tmp_path)

    assert [item.candidate_id for item in selected] == ["second"]
    assert reference_llm is None


def test_resolve_last_run_selection_reports_stale_model_id(tmp_path: Path):
    config = {
        "llm_reference": {"custom_model_paths": []},
        "last_run_selection": {"schema": LAST_RUN_SELECTION_SCHEMA, "candidate_ids": ["missing"], "reference_llm_candidate_id": ""},
    }

    selected, reference_llm, errors = resolve_last_run_selection([candidate("available", tmp_path)], [], config)

    assert selected == []
    assert reference_llm is None
    assert errors == ["saved ASR model id not found: missing"]


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
