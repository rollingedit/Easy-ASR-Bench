from pathlib import Path

from app.llm_reference import merge_reference_llms, save_custom_reference_path, scan_custom_reference_llms


def test_save_custom_reference_file_path_and_rescan(tmp_path: Path):
    config_path = tmp_path / "config.json"
    model_path = tmp_path / "other_app" / "models" / "reference.Q4_K_M.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"gguf")
    config = {"llm_reference": {"custom_model_paths": []}}

    candidates = save_custom_reference_path(config_path, config, str(model_path))

    assert len(candidates) == 1
    assert config["llm_reference"]["custom_model_paths"] == [str(model_path.resolve())]
    rescanned = scan_custom_reference_llms(config)
    assert [candidate.path for candidate in rescanned] == [model_path]


def test_save_custom_reference_folder_path(tmp_path: Path):
    folder = tmp_path / "lm_studio_models"
    folder.mkdir()
    (folder / "first.gguf").write_bytes(b"gguf")
    (folder / "second.gguf").write_bytes(b"gguf")
    config = {"llm_reference": {"custom_model_paths": []}}

    candidates = save_custom_reference_path(tmp_path / "config.json", config, str(folder))

    assert sorted(candidate.display_name for candidate in candidates) == ["first.gguf", "second.gguf"]


def test_merge_reference_llms_deduplicates_by_resolved_path(tmp_path: Path):
    model_path = tmp_path / "reference.gguf"
    model_path.write_bytes(b"gguf")
    config = {"llm_reference": {"custom_model_paths": []}}
    first = save_custom_reference_path(tmp_path / "config.json", config, str(model_path))
    second = scan_custom_reference_llms(config)

    merged = merge_reference_llms(first, second)

    assert len(merged) == 1
