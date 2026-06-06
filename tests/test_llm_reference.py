from pathlib import Path

from app.llm_reference import merge_reference_llms, save_custom_reference_path, scan_custom_reference_llms
from app.precision_detector import detect_from_path, normalize_precision_label


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


def test_gguf_quantization_labels_are_detected(tmp_path: Path):
    names = {
        "model.Q4_K_M.gguf": "4-bit / Q4",
        "model.Q5_K_S.gguf": "5-bit / Q5",
        "model.Q6_K.gguf": "6-bit / Q6",
        "model.IQ3_XXS.gguf": "3-bit / IQ3",
        "model.FP8.gguf": "8-bit / INT8 / Q8",
        "model.BF16.gguf": "16-bit / BF16",
        "model.BF8.gguf": "8-bit / INT8 / Q8",
        "model.NF4.gguf": "4-bit / Q4",
        "model.NVFP4.gguf": "4-bit / Q4",
    }

    for name, bucket in names.items():
        path = tmp_path / name
        path.write_bytes(b"gguf")
        _, detected_bucket = detect_from_path(path)
        assert detected_bucket == bucket


def test_precision_normalization_covers_common_asr_quant_labels():
    for label, bucket in {
        "int4": "4-bit / Q4",
        "int5": "5-bit / Q5",
        "int6": "6-bit / Q6",
        "fp4": "4-bit / Q4",
        "fp8": "8-bit / INT8 / Q8",
        "bf16": "16-bit / BF16",
        "bfloat16": "16-bit / BF16",
        "bf8": "8-bit / INT8 / Q8",
        "nf4": "4-bit / Q4",
        "nvfp4": "4-bit / Q4",
        "nvp4": "4-bit / Q4",
    }.items():
        assert normalize_precision_label(label)[1] == bucket
