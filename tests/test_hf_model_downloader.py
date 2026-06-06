import json
from pathlib import Path

from app.adapters.base import ModelCandidate
from app.hf_model_downloader import (
    DownloadChoice,
    HFModelRef,
    build_download_choices,
    build_smart_download_choices,
    destination_for,
    download_hf_model_interactive,
    download_choice,
    local_relative_name,
    offer_missing_file_repair,
    parent_refs,
    parse_hf_model_ref,
)


def test_parse_hf_model_ref_from_repo_url_with_subfolder():
    ref = parse_hf_model_ref("https://huggingface.co/owner/model/tree/main/onnx/int8")

    assert ref.repo_id == "owner/model"
    assert ref.revision == "main"
    assert ref.subfolder == "onnx/int8"


def test_parse_hf_model_ref_accepts_granite_repo_url_shapes():
    base = "https://huggingface.co/ibm-granite/granite-4.0-1b-speech"

    assert parse_hf_model_ref(base).repo_id == "ibm-granite/granite-4.0-1b-speech"
    assert parse_hf_model_ref(base + "/tree/main/").subfolder == ""
    assert parse_hf_model_ref(base + "/tree/main/some/nested/folder///").subfolder == "some/nested/folder"


def test_parse_blob_link_uses_parent_folder():
    ref = parse_hf_model_ref("https://huggingface.co/owner/model/blob/main/onnx/int8/encoder.onnx")

    assert ref.repo_id == "owner/model"
    assert ref.revision == "main"
    assert ref.subfolder == "onnx/int8"


def test_parent_refs_walk_to_repo_root():
    refs = parent_refs(HFModelRef("owner/model", "main", "onnx/int8/some_nested_folder"))

    assert [ref.subfolder for ref in refs] == ["onnx/int8/some_nested_folder", "onnx/int8", "onnx", ""]


def test_smart_choices_fall_back_from_arbitrary_nested_folder_to_parent_package():
    files = [
        "onnx/int8/some_nested_folder/scores.json",
        "onnx/int8/encoder.onnx",
        "onnx/int8/decoder_model_merged.onnx",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ]

    ref, choices = build_smart_download_choices(files, HFModelRef("owner/model", "main", "onnx/int8/some_nested_folder"))

    assert ref.subfolder == "onnx/int8"
    choice = next(choice for choice in choices if choice.kind == "onnx")
    assert "config.json" in choice.files
    assert "tokenizer.json" in choice.files
    assert "preprocessor_config.json" in choice.files


def test_build_choices_does_not_download_every_safetensors_weight():
    files = [
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "model.fp16.safetensors",
        "model.int8.safetensors",
    ]

    choices = build_download_choices(files, HFModelRef("owner/asr"))

    safetensor_choices = [choice for choice in choices if choice.kind == "safetensors"]
    assert len(safetensor_choices) == 2
    assert all(not {"model.fp16.safetensors", "model.int8.safetensors"} <= set(choice.files) for choice in safetensor_choices)
    assert all("config.json" in choice.files for choice in safetensor_choices)


def test_build_choices_warns_when_required_metadata_is_missing():
    files = ["model.safetensors"]

    choices = build_download_choices(files, HFModelRef("owner/asr"))

    choice = choices[0]
    assert any("config.json" in note for note in choice.notes)
    assert any("tokenizer" in note for note in choice.notes)
    assert any("processor" in note for note in choice.notes)


def test_build_choices_groups_split_safetensors_parts_as_one_choice():
    files = [
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "model.fp32-00001-of-00002.safetensors",
        "model.fp32-00002-of-00002.safetensors",
        "model.safetensors",
        "model.safetensors.index.fp32.json",
    ]

    choices = build_download_choices(files, HFModelRef("openai/whisper-large-v3"))

    indexed = next(choice for choice in choices if choice.primary_files == ("model.safetensors.index.fp32.json",))
    assert "Shard files will be read" in indexed.notes[0]
    assert "model.safetensors.index.fp32.json" in indexed.files
    assert any(choice.primary_files == ("model.safetensors",) for choice in choices)


def test_build_choices_groups_split_safetensors_without_index():
    files = [
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "model.fp32-00001-of-00002.safetensors",
        "model.fp32-00002-of-00002.safetensors",
    ]

    choices = build_download_choices(files, HFModelRef("owner/asr"))

    split = next(choice for choice in choices if "split Safetensors" in choice.label)
    assert split.files == (
        "config.json",
        "model.fp32-00001-of-00002.safetensors",
        "model.fp32-00002-of-00002.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
    )


def test_build_choices_pairs_asr_gguf_with_matching_projector():
    files = [
        "Qwen3-ASR-1.7B-Q8_0.gguf",
        "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
        "Qwen3-ASR-1.7B-Q4_K_M.gguf",
    ]

    choices = build_download_choices(files, HFModelRef("owner/qwen-asr"))

    q8 = next(choice for choice in choices if "Q8_0" in choice.label)
    assert q8.files == ("Qwen3-ASR-1.7B-Q8_0.gguf", "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")
    assert q8.task_hint == "asr_audio"
    q4 = next(choice for choice in choices if "Q4_K_M" in choice.label)
    assert "No matching mmproj" in q4.notes[0]


def test_build_choices_marks_regular_gguf_as_reference_llm():
    files = ["BF16/Qwen3.5-27B-BF16.gguf", "Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf"]

    ref, choices = build_smart_download_choices(files, HFModelRef("unsloth/Qwen3.5-27B-GGUF", "main", "BF16"))

    assert ref.subfolder == "BF16"
    assert len(choices) == 1
    assert choices[0].task_hint == "reference_llm"
    assert choices[0].files == ("BF16/Qwen3.5-27B-BF16.gguf",)


def test_build_choices_groups_split_gguf_parts_as_one_llm_choice():
    files = [
        "BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf",
        "BF16/Qwen3.5-27B-BF16-00002-of-00002.gguf",
        "Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf",
    ]

    ref, choices = build_smart_download_choices(files, HFModelRef("unsloth/Qwen3.5-27B-GGUF", "main", "BF16"))

    assert ref.subfolder == "BF16"
    assert len(choices) == 1
    assert choices[0].label == "GGUF reference LLM: Qwen3.5-27B-BF16 split GGUF"
    assert choices[0].files == (
        "BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf",
        "BF16/Qwen3.5-27B-BF16-00002-of-00002.gguf",
    )


def test_build_choices_groups_onnx_package_with_sidecars_not_other_dirs():
    files = [
        "config.json",
        "preprocessor_config.json",
        "onnx/int8/encoder.onnx",
        "onnx/int8/encoder.onnx_data",
        "onnx/int8/decoder_model_merged.onnx",
        "onnx/fp16/encoder.onnx",
        "model.safetensors",
    ]

    choices = build_download_choices(files, HFModelRef("owner/asr"))

    int8 = next(choice for choice in choices if choice.kind == "onnx" and "onnx/int8" in choice.label)
    assert "onnx/int8/encoder.onnx" in int8.files
    assert "onnx/int8/encoder.onnx_data" in int8.files
    assert "config.json" in int8.files
    assert "preprocessor_config.json" in int8.files
    assert "onnx/fp16/encoder.onnx" not in int8.files
    assert "model.safetensors" not in int8.files


def test_build_choices_splits_onnx_variants_inside_one_folder():
    files = [
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "onnx/audio_encoder.onnx",
        "onnx/audio_encoder.onnx_data",
        "onnx/audio_encoder_fp16.onnx",
        "onnx/audio_encoder_fp16.onnx_data",
        "onnx/audio_encoder_q4.onnx",
        "onnx/audio_encoder_q4.onnx_data",
        "onnx/decoder_model_merged.onnx",
        "onnx/decoder_model_merged.onnx_data",
        "onnx/decoder_model_merged_fp16.onnx",
        "onnx/decoder_model_merged_fp16.onnx_data",
        "onnx/decoder_model_merged_q4.onnx",
        "onnx/decoder_model_merged_q4.onnx_data",
        "onnx/embed_tokens.onnx",
        "onnx/embed_tokens.onnx_data",
        "onnx/embed_tokens_fp16.onnx",
        "onnx/embed_tokens_fp16.onnx_data",
        "onnx/embed_tokens_q4.onnx",
        "onnx/embed_tokens_q4.onnx_data",
    ]

    choices = build_download_choices(files, HFModelRef("onnx-community/granite"))

    default = next(choice for choice in choices if choice.kind == "onnx" and "[default]" in choice.label)
    fp16 = next(choice for choice in choices if choice.kind == "onnx" and "[fp16]" in choice.label)
    q4 = next(choice for choice in choices if choice.kind == "onnx" and "[q4]" in choice.label)
    assert "onnx/audio_encoder_fp16.onnx" not in default.files
    assert "onnx/audio_encoder_fp16.onnx_data" not in default.files
    assert "onnx/audio_encoder.onnx" not in fp16.files
    assert "onnx/audio_encoder.onnx_data" not in fp16.files
    assert "onnx/audio_encoder_q4.onnx" in q4.files
    assert "onnx/decoder_model_merged_q4.onnx_data" in q4.files


def test_build_choices_keeps_root_int8_onnx_with_tokens():
    files = [
        "cohere-decoder.int8.onnx",
        "cohere-encoder.int8.onnx",
        "cohere-encoder.int8.onnx.data",
        "tokens.txt",
    ]

    choices = build_download_choices(files, HFModelRef("owner/cohere-onnx"))

    int8 = next(choice for choice in choices if choice.kind == "onnx" and "[int8]" in choice.label)
    assert int8.files == (
        "cohere-decoder.int8.onnx",
        "cohere-encoder.int8.onnx",
        "cohere-encoder.int8.onnx.data",
        "tokens.txt",
    )


def test_smart_choices_offer_unknown_subfolder_for_inspection_only():
    files = [
        "experimental/export.foo",
        "experimental/weights.custom",
        "config.json",
        "tokenizer.json",
    ]

    ref, choices = build_smart_download_choices(files, HFModelRef("owner/model", "main", "experimental"))

    assert ref.subfolder == "experimental"
    assert len(choices) == 1
    assert choices[0].kind == "folder"
    assert choices[0].task_hint == "unknown"
    assert "experimental/export.foo" in choices[0].files
    assert "config.json" in choices[0].files
    assert any("not treated as runnable" in note for note in choices[0].notes)


def test_download_choice_expands_sharded_safetensors_index(tmp_path: Path, monkeypatch):
    def fake_download(repo_id: str, revision: str | None, filename: str, destination: Path, relative_name: str | None = None) -> Path:
        path = destination / (relative_name or filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename == "model.safetensors.index.json":
            path.write_text(
                json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}),
                encoding="utf-8",
            )
        else:
            path.write_text("", encoding="utf-8")
        return path

    monkeypatch.setattr("app.hf_model_downloader._download_file", fake_download)
    choice = DownloadChoice(
        label="Sharded Safetensors",
        kind="safetensors_index",
        primary_files=("model.safetensors.index.json",),
        files=("config.json", "model.safetensors.index.json"),
    )

    downloaded = download_choice(HFModelRef("owner/asr"), choice, tmp_path, print_func=lambda _: None)

    assert tmp_path / "model-00001-of-00002.safetensors" in downloaded
    assert tmp_path / "model-00002-of-00002.safetensors" in downloaded


def test_download_choice_skips_existing_local_files(tmp_path: Path, monkeypatch):
    (tmp_path / "config.json").write_text("already here", encoding="utf-8")
    downloaded: list[str] = []

    def fake_download(repo_id: str, revision: str | None, filename: str, destination: Path, relative_name: str | None = None) -> Path:
        downloaded.append(filename)
        path = destination / (relative_name or filename)
        path.write_text("", encoding="utf-8")
        return path

    monkeypatch.setattr("app.hf_model_downloader._download_file", fake_download)
    choice = DownloadChoice(
        label="Safetensors",
        kind="safetensors",
        primary_files=("model.safetensors",),
        files=("config.json", "model.safetensors"),
    )

    download_choice(HFModelRef("owner/asr"), choice, tmp_path, print_func=lambda _: None)

    assert downloaded == ["model.safetensors"]
    assert (tmp_path / "config.json").read_text(encoding="utf-8") == "already here"


def test_download_choice_expands_nested_sharded_safetensors_into_one_folder(tmp_path: Path, monkeypatch):
    def fake_download(repo_id: str, revision: str | None, filename: str, destination: Path, relative_name: str | None = None) -> Path:
        path = destination / (relative_name or filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename == "BF16/model.safetensors.index.json":
            path.write_text(
                json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}),
                encoding="utf-8",
            )
        else:
            path.write_text("", encoding="utf-8")
        return path

    monkeypatch.setattr("app.hf_model_downloader._download_file", fake_download)
    choice = DownloadChoice(
        label="Sharded Safetensors",
        kind="safetensors_index",
        primary_files=("BF16/model.safetensors.index.json",),
        files=("config.json", "BF16/model.safetensors.index.json"),
        task_hint="metadata_required",
    )

    downloaded = download_choice(HFModelRef("owner/asr", subfolder="BF16"), choice, tmp_path, print_func=lambda _: None)

    assert tmp_path / "model.safetensors.index.json" in downloaded
    assert tmp_path / "model-00001-of-00002.safetensors" in downloaded
    assert tmp_path / "model-00002-of-00002.safetensors" in downloaded
    assert not (tmp_path / "BF16").exists()


def test_destination_for_is_stable_and_readable(tmp_path: Path):
    destination = destination_for(tmp_path, HFModelRef("owner/model", subfolder="onnx/int8"))

    assert destination == tmp_path / "owner__model__onnx_int8"


def test_destination_for_choice_uses_one_package_folder(tmp_path: Path):
    choice = DownloadChoice(
        label="GGUF reference LLM",
        kind="gguf",
        primary_files=("BF16/Qwen3.5-27B-BF16.gguf",),
        files=("BF16/Qwen3.5-27B-BF16.gguf",),
        task_hint="reference_llm",
    )

    destination = destination_for(tmp_path, HFModelRef("unsloth/Qwen3.5-27B-GGUF", "main", "BF16"), choice)

    assert destination == tmp_path / "unsloth__Qwen3.5-27B-GGUF__Qwen3.5-27B-BF16"
    assert local_relative_name(choice, "BF16/Qwen3.5-27B-BF16.gguf") == "Qwen3.5-27B-BF16.gguf"


def test_destination_for_split_gguf_removes_part_suffix(tmp_path: Path):
    choice = DownloadChoice(
        label="GGUF reference LLM",
        kind="gguf",
        primary_files=("BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf",),
        files=("BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf", "BF16/Qwen3.5-27B-BF16-00002-of-00002.gguf"),
        task_hint="reference_llm",
    )

    destination = destination_for(tmp_path, HFModelRef("unsloth/Qwen3.5-27B-GGUF", "main", "BF16"), choice)

    assert destination == tmp_path / "unsloth__Qwen3.5-27B-GGUF__Qwen3.5-27B-BF16"


def test_interactive_download_requires_confirmation_for_large_choice(tmp_path: Path, monkeypatch):
    files = ["config.json", "tokenizer.json", "preprocessor_config.json"]
    files.extend(f"onnx/encoder_model_{index}.onnx" for index in range(21))
    monkeypatch.setattr("app.hf_model_downloader.list_repo_files", lambda ref: files)
    called = {"downloaded": False}

    def fake_download(ref, choice, destination, print_func=print):
        called["downloaded"] = True
        return []

    monkeypatch.setattr("app.hf_model_downloader.download_choice", fake_download)
    answers = iter(["owner/model", "n"])

    result = download_hf_model_interactive(tmp_path, input_func=lambda prompt: next(answers), print_func=lambda text: None)

    assert result is None
    assert called["downloaded"] is False


def test_offer_missing_file_repair_downloads_exact_repo_matches(tmp_path: Path, monkeypatch):
    destination = tmp_path / "model"
    destination.mkdir()
    (destination / "model.safetensors").write_text("", encoding="utf-8")
    repo_files = ["model.safetensors", "config.json", "preprocessor_config.json", "tokenizer.json"]
    choice = DownloadChoice(
        label="Safetensors",
        kind="safetensors",
        primary_files=("model.safetensors",),
        files=("model.safetensors",),
        task_hint="metadata_required",
    )
    downloaded: list[str] = []

    def fake_download(repo_id: str, revision: str | None, filename: str, destination: Path, relative_name: str | None = None) -> Path:
        downloaded.append(filename)
        path = destination / (relative_name or filename)
        path.write_text("", encoding="utf-8")
        return path

    monkeypatch.setattr("app.hf_model_downloader._download_file", fake_download)
    missing_candidate = ModelCandidate(
        candidate_id="incomplete",
        display_name="Incomplete",
        family_name="Incomplete",
        backend="transformers",
        container_format="safetensors",
        task="unknown",
        precision="unknown",
        quantization_label="Unknown precision",
        path=destination,
        adapter_name="hf_transformers_asr",
        runnable=False,
        missing_files=["config.json", "preprocessor_config.json"],
    )
    monkeypatch.setattr("app.model_scanner.scan_models", lambda root: ([], [missing_candidate]))

    offer_missing_file_repair(
        HFModelRef("owner/asr"),
        choice,
        repo_files,
        destination,
        input_func=lambda prompt: "y",
        print_func=lambda text: None,
    )

    assert "config.json" in downloaded
    assert "preprocessor_config.json" in downloaded


def test_offer_missing_file_repair_writes_structured_llm_audit_request_when_ambiguous(tmp_path: Path, monkeypatch):
    destination = tmp_path / "model"
    destination.mkdir()
    repo_files = ["model.safetensors", "config.json", "tokenizer.json"]
    choice = DownloadChoice(
        label="Safetensors",
        kind="safetensors",
        primary_files=("model.safetensors",),
        files=("model.safetensors",),
        task_hint="metadata_required",
    )
    missing_candidate = ModelCandidate(
        candidate_id="incomplete",
        display_name="Incomplete",
        family_name="Incomplete",
        backend="transformers",
        container_format="safetensors",
        task="unknown",
        precision="unknown",
        quantization_label="Unknown precision",
        path=destination,
        adapter_name="hf_transformers_asr",
        runnable=False,
        missing_files=["tokenizer/vocab file"],
    )
    monkeypatch.setattr("app.model_scanner.scan_models", lambda root: ([], [missing_candidate]))

    offer_missing_file_repair(
        HFModelRef("owner/asr"),
        choice,
        repo_files,
        destination,
        input_func=lambda prompt: "y",
        print_func=lambda text: None,
    )

    request_path = destination / "hf_missing_file_request.json"
    prompt_path = destination / "hf_missing_file_prompt.txt"
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "easy_asr_bench.hf_missing_file_request.v1"
    assert payload["repo_files"] == repo_files
    assert payload["scanner_missing_files"] == ["tokenizer/vocab file"]
    assert payload["required_response_schema"]["schema"] == "easy_asr_bench.hf_missing_file_recommendation.v1"
    assert "Return only JSON" in prompt_path.read_text(encoding="utf-8")
