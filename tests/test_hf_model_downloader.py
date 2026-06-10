import json
from pathlib import Path

from app.adapters.base import ModelCandidate
from app.hf_model_downloader import (
    DownloadChoice,
    HFModelRef,
    RECOMMENDED_BASELINE_REPO,
    RepoInspection,
    build_missing_file_repair_plan,
    build_download_choices,
    build_smart_download_choices,
    destination_for,
    download_hf_model_interactive,
    download_choice,
    execute_missing_file_repair_plan,
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


def test_build_choices_detects_faster_whisper_ctranslate2_package():
    files = [
        "config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.txt",
        "README.md",
    ]

    choices = build_download_choices(files, HFModelRef(RECOMMENDED_BASELINE_REPO))

    choice = next(choice for choice in choices if choice.kind == "ctranslate2")
    assert choice.task_hint == "asr_audio"
    assert set(choice.files) == {"config.json", "model.bin", "tokenizer.json", "vocabulary.txt"}


def test_faster_whisper_choice_requires_runnable_metadata():
    files = ["model.bin", "README.md"]

    choices = build_download_choices(files, HFModelRef("owner/incomplete-faster-whisper"))

    assert not any(choice.kind == "ctranslate2" for choice in choices)


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


def test_build_choices_pairs_real_qwen_infix_mmproj_projector():
    files = [
        "Qwen3-ASR-0.6B.Q4_K_M.gguf",
        "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf",
        "Qwen3-ASR-0.6B.Q2_K.gguf",
    ]

    choices = build_download_choices(files, HFModelRef("mradermacher/Qwen3-ASR-0.6B-GGUF"))

    q4 = next(choice for choice in choices if "Q4_K_M" in choice.label)
    assert q4.task_hint == "asr_audio"
    assert q4.files == ("Qwen3-ASR-0.6B.Q4_K_M.gguf", "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf")
    assert "mmproj-Q8_0" in q4.label


def test_build_choices_splits_real_qwen_multiple_projectors_into_exact_pairs():
    files = [
        "Qwen3-ASR-0.6B.Q4_K_M.gguf",
        "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf",
        "Qwen3-ASR-0.6B.mmproj-f16.gguf",
    ]

    choices = build_download_choices(files, HFModelRef("mradermacher/Qwen3-ASR-0.6B-GGUF"))

    q4_choices = [choice for choice in choices if "Q4_K_M" in choice.label and choice.task_hint == "asr_audio"]
    assert len(q4_choices) == 2
    assert {choice.files for choice in q4_choices} == {
        ("Qwen3-ASR-0.6B.Q4_K_M.gguf", "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf"),
        ("Qwen3-ASR-0.6B.Q4_K_M.gguf", "Qwen3-ASR-0.6B.mmproj-f16.gguf"),
    }


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


def test_download_choice_writes_manifest_for_real_qwen_asr_gguf_pair(tmp_path: Path, monkeypatch):
    def fake_download(repo_id: str, revision: str | None, filename: str, destination: Path, relative_name: str | None = None) -> Path:
        path = destination / (relative_name or filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"gguf")
        return path

    monkeypatch.setattr("app.hf_model_downloader._download_file", fake_download)
    choice = DownloadChoice(
        label="Audio/ASR GGUF: Qwen3-ASR-0.6B.Q4_K_M + Qwen3-ASR-0.6B.mmproj-Q8_0.gguf",
        kind="gguf",
        primary_files=("Qwen3-ASR-0.6B.Q4_K_M.gguf",),
        files=("Qwen3-ASR-0.6B.Q4_K_M.gguf", "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf"),
        task_hint="asr_audio",
    )

    downloaded = download_choice(HFModelRef("mradermacher/Qwen3-ASR-0.6B-GGUF"), choice, tmp_path, print_func=lambda _: None)

    manifest_path = tmp_path / "model_package.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path in downloaded
    assert payload["artifacts"] == {
        "main_model": "Qwen3-ASR-0.6B.Q4_K_M.gguf",
        "projector": "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf",
    }


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


def test_destination_for_bounds_long_repo_and_package_names(tmp_path: Path):
    choice = DownloadChoice(
        label="Audio/ASR GGUF",
        kind="gguf",
        primary_files=("Qwen3-ASR-0.6B.Q4_K_M.gguf",),
        files=("Qwen3-ASR-0.6B.Q4_K_M.gguf", "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf"),
        task_hint="asr_audio",
    )

    destination = destination_for(
        tmp_path,
        HFModelRef("very-long-owner-name-for-windows-path-testing/Qwen3-ASR-0.6B-GGUF-with-extra-long-repo-suffix"),
        choice,
    )

    assert len(destination.name) <= 96
    assert destination.name.startswith("very-long-owner-name")
    assert "Qwen3-ASR-0.6B.Q4_K_M" in destination.name


def test_destination_for_shrinks_further_under_deep_roots(tmp_path: Path):
    choice = DownloadChoice(
        label="Audio/ASR GGUF",
        kind="gguf",
        primary_files=("Qwen3-ASR-0.6B.Q4_K_M.gguf",),
        files=("Qwen3-ASR-0.6B.Q4_K_M.gguf", "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf"),
        task_hint="asr_audio",
    )
    deep_root = tmp_path / ("deep_path_segment_" * 4) / "Models"

    destination = destination_for(deep_root, HFModelRef("mradermacher/Qwen3-ASR-0.6B-GGUF"), choice)
    longest_target = destination / "Qwen3-ASR-0.6B.mmproj-Q8_0.gguf"

    assert len(str(longest_target if longest_target.is_absolute() else Path.cwd() / longest_target)) <= 240


def test_interactive_download_requires_confirmation_for_large_choice(tmp_path: Path, monkeypatch):
    files = ["config.json", "tokenizer.json", "preprocessor_config.json"]
    files.extend(f"onnx/encoder_model_{index}.onnx" for index in range(21))
    sizes = {filename: 120 * 1024 * 1024 for filename in files}
    monkeypatch.setattr("app.hf_model_downloader.inspect_repo", lambda ref: RepoInspection(ref, files, sizes))
    called = {"downloaded": False}

    def fake_download(ref, choice, destination, print_func=print):
        called["downloaded"] = True
        return []

    monkeypatch.setattr("app.hf_model_downloader.download_choice", fake_download)
    answers = iter(["owner/model", "n", ""])

    result = download_hf_model_interactive(tmp_path, input_func=lambda prompt: next(answers), print_func=lambda text: None)

    assert result is None
    assert called["downloaded"] is False


def test_interactive_download_rejects_y_for_large_choice(tmp_path: Path, monkeypatch):
    files = ["config.json", "tokenizer.json", "preprocessor_config.json"]
    files.extend(f"onnx/encoder_model_{index}.onnx" for index in range(21))
    sizes = {filename: 120 * 1024 * 1024 for filename in files}
    monkeypatch.setattr("app.hf_model_downloader.inspect_repo", lambda ref: RepoInspection(ref, files, sizes))
    called = {"downloaded": False}

    def fake_download(ref, choice, destination, print_func=print):
        called["downloaded"] = True
        return []

    monkeypatch.setattr("app.hf_model_downloader.download_choice", fake_download)
    answers = iter(["owner/model", "y", ""])

    result = download_hf_model_interactive(tmp_path, input_func=lambda prompt: next(answers), print_func=lambda text: None)

    assert result is None
    assert called["downloaded"] is False


def test_interactive_download_accepts_typed_download_for_large_choice(tmp_path: Path, monkeypatch):
    files = ["config.json", "tokenizer.json", "preprocessor_config.json"]
    files.extend(f"onnx/encoder_model_{index}.onnx" for index in range(21))
    sizes = {filename: 120 * 1024 * 1024 for filename in files}
    monkeypatch.setattr("app.hf_model_downloader.inspect_repo", lambda ref: RepoInspection(ref, files, sizes))
    called = {"downloaded": False}

    def fake_download(ref, choice, destination, print_func=print):
        called["downloaded"] = True
        return []

    monkeypatch.setattr("app.hf_model_downloader.download_choice", fake_download)
    answers = iter(["owner/model", "DOWNLOAD", ""])

    result = download_hf_model_interactive(tmp_path, input_func=lambda prompt: next(answers), print_func=lambda text: None)

    assert result is not None
    assert called["downloaded"] is True


def test_interactive_download_inspection_failure_returns_to_user_without_crash(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("app.hf_model_downloader.list_repo_files", lambda ref: (_ for _ in ()).throw(RuntimeError("network down")))
    messages: list[str] = []
    answers = iter(["owner/model", ""])

    result = download_hf_model_interactive(
        tmp_path,
        input_func=lambda prompt: next(answers),
        print_func=messages.append,
    )

    assert result is None
    assert any("Could not inspect that Hugging Face model" in message for message in messages)
    assert any("network down" in message for message in messages)


def test_interactive_download_gated_repo_error_prints_token_guidance(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("app.hf_model_downloader.inspect_repo", lambda ref: (_ for _ in ()).throw(RuntimeError("403 gated repo")))
    messages: list[str] = []
    answers = iter(["owner/private", ""])

    result = download_hf_model_interactive(tmp_path, input_func=lambda prompt: next(answers), print_func=messages.append)

    assert result is None
    assert any("huggingface-cli login" in message for message in messages)
    assert any("accept the model terms" in message for message in messages)


def test_interactive_download_eof_returns_without_crash(tmp_path: Path):
    messages: list[str] = []

    result = download_hf_model_interactive(
        tmp_path,
        input_func=lambda prompt: (_ for _ in ()).throw(EOFError()),
        print_func=messages.append,
    )

    assert result is None
    assert any("Input closed" in message for message in messages)


def test_interactive_download_loops_until_blank_enter(tmp_path: Path, monkeypatch):
    files = ["config.json", "tokenizer.json", "preprocessor_config.json", "model.safetensors"]
    monkeypatch.setattr("app.hf_model_downloader.list_repo_files", lambda ref: files)
    downloaded: list[str] = []

    def fake_download(ref, choice, destination, print_func=print):
        downloaded.append(ref.repo_id)
        destination.mkdir(parents=True, exist_ok=True)
        return [destination / "model.safetensors"]

    monkeypatch.setattr("app.hf_model_downloader.download_choice", fake_download)
    answers = iter(["bad link", "owner/first", "owner/second", ""])
    messages: list[str] = []

    result = download_hf_model_interactive(tmp_path, input_func=lambda prompt: next(answers), print_func=messages.append)

    assert downloaded == ["owner/first", "owner/second"]
    assert result == destination_for(tmp_path, HFModelRef("owner/second"), build_smart_download_choices(files, HFModelRef("owner/second"))[1][0])
    assert any("Invalid Hugging Face link or repo id" in message for message in messages)
    assert any("Paste another Hugging Face link, or press Enter when done." in message for message in messages)


def test_download_writes_resume_plan_before_first_file_and_prints_doctor_command(tmp_path: Path, monkeypatch):
    files = ["config.json", "tokenizer.json", "preprocessor_config.json", "model.safetensors"]
    monkeypatch.setattr("app.hf_model_downloader.list_repo_files", lambda ref: files)
    messages: list[str] = []

    def fake_download(ref, choice, destination, print_func=print):
        raise RuntimeError("connection reset")

    monkeypatch.setattr("app.hf_model_downloader.download_choice", fake_download)
    answers = iter(["owner/asr", ""])

    result = download_hf_model_interactive(tmp_path, input_func=lambda prompt: next(answers), print_func=messages.append)

    destination = destination_for(tmp_path, HFModelRef("owner/asr"), build_smart_download_choices(files, HFModelRef("owner/asr"))[1][0])
    plan_path = destination / "hf_model_layout_repair_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert result is None
    assert plan["schema"] == "easy_asr_bench.model_layout_repair_plan.v1"
    assert plan["records"][0]["repair_action"] == "download_selected_hf_package"
    assert set(plan["records"][0]["safe_download_files"]) == set(files)
    assert any("setup.bat --doctor --repair-model-layouts --allow-downloads" in message for message in messages)
    assert any("Download failed" in message for message in messages)


def test_download_uses_resolved_hub_revision_for_materialization(tmp_path: Path, monkeypatch):
    files = ["config.json", "tokenizer.json", "preprocessor_config.json", "model.safetensors"]
    resolved_ref = HFModelRef("owner/asr", "0123456789abcdef0123456789abcdef01234567")
    monkeypatch.setattr("app.hf_model_downloader.inspect_repo", lambda ref: RepoInspection(resolved_ref, files, {"model.safetensors": 123}))
    revisions: list[str | None] = []

    def fake_download(ref, choice, destination, print_func=print):
        revisions.append(ref.revision)
        destination.mkdir(parents=True, exist_ok=True)
        return [destination / "model.safetensors"]

    monkeypatch.setattr("app.hf_model_downloader.download_choice", fake_download)
    answers = iter(["owner/asr", ""])

    result = download_hf_model_interactive(tmp_path, input_func=lambda prompt: next(answers), print_func=lambda text: None)

    plan = json.loads((result / "hf_model_layout_repair_plan.json").read_text(encoding="utf-8"))
    assert revisions == [resolved_ref.revision]
    assert plan["revision"] == resolved_ref.revision


def test_interactive_download_unknown_choice_requires_confirmation(tmp_path: Path, monkeypatch):
    files = ["custom/export.foo", "custom/weights.custom"]
    monkeypatch.setattr("app.hf_model_downloader.list_repo_files", lambda ref: files)
    called = {"downloaded": False}

    def fake_download(ref, choice, destination, print_func=print):
        called["downloaded"] = True
        return []

    monkeypatch.setattr("app.hf_model_downloader.download_choice", fake_download)
    answers = iter(["https://huggingface.co/owner/model/tree/main/custom", "n", ""])
    messages: list[str] = []

    result = download_hf_model_interactive(tmp_path, input_func=lambda prompt: next(answers), print_func=messages.append)

    assert result is None
    assert called["downloaded"] is False
    assert any("unknown package layout" in message for message in messages)


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

    repair_plan_path = destination / "hf_model_layout_repair_plan.json"
    repair_plan = json.loads(repair_plan_path.read_text(encoding="utf-8"))
    assert repair_plan["schema"] == "easy_asr_bench.model_layout_repair_plan.v1"
    assert repair_plan["records"][0]["safe_download_files"] == ["config.json", "preprocessor_config.json"]
    assert repair_plan["last_execution"]["schema"] == "easy_asr_bench.model_layout_repair_execution.v1"
    assert repair_plan["last_execution"]["summary"]["repaired"] == 1
    assert "config.json" in downloaded
    assert "preprocessor_config.json" in downloaded


def test_build_missing_file_repair_plan_records_exact_sidecars(tmp_path: Path):
    destination = tmp_path / "model"
    destination.mkdir()
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
        missing_files=["config.json", "preprocessor_config.json"],
    )

    plan = build_missing_file_repair_plan(
        HFModelRef("owner/asr"),
        choice,
        ["model.safetensors", "config.json", "preprocessor_config.json", "tokenizer.json"],
        destination,
        [missing_candidate],
    )

    record = plan["records"][0]
    assert plan["schema"] == "easy_asr_bench.model_layout_repair_plan.v1"
    assert plan["summary"] == {"total": 1, "needs_repair": 1, "can_auto_repair": 1, "blocked": 0}
    assert record["issue_id"] == "model_layout:incomplete"
    assert record["repair_action"] == "download_exact_missing_files"
    assert record["can_auto_repair"] is True
    assert record["requires_confirmation"] is True
    assert record["safe_download_files"] == ["config.json", "preprocessor_config.json"]


def test_build_missing_file_repair_plan_records_audit_blocker(tmp_path: Path):
    destination = tmp_path / "model"
    destination.mkdir()
    choice = DownloadChoice(
        label="Safetensors",
        kind="safetensors",
        primary_files=("model.safetensors",),
        files=("model.safetensors",),
        task_hint="metadata_required",
    )
    missing_candidate = ModelCandidate(
        candidate_id="ambiguous",
        display_name="Ambiguous",
        family_name="Ambiguous",
        backend="transformers",
        container_format="safetensors",
        task="unknown",
        precision="unknown",
        quantization_label="Unknown precision",
        path=destination,
        adapter_name="hf_transformers_asr",
        runnable=False,
        missing_files=["custom processor sidecar"],
    )

    plan = build_missing_file_repair_plan(
        HFModelRef("owner/asr"),
        choice,
        ["model.safetensors", "README.md"],
        destination,
        [missing_candidate],
    )

    record = plan["records"][0]
    assert plan["summary"]["blocked"] == 1
    assert record["repair_action"] == "write_llm_file_audit_request"
    assert record["can_auto_repair"] is False
    assert "No exact or conservative" in record["block_reason"]


def test_execute_missing_file_repair_plan_downloads_allowed_sidecars(tmp_path: Path, monkeypatch):
    destination = tmp_path / "model"
    destination.mkdir()
    choice = DownloadChoice(
        label="Safetensors",
        kind="safetensors",
        primary_files=("model.safetensors",),
        files=("model.safetensors",),
        task_hint="metadata_required",
    )
    plan = {
        "schema": "easy_asr_bench.model_layout_repair_plan.v1",
        "records": [
            {
                "issue_id": "model_layout:incomplete",
                "repair_action": "download_exact_missing_files",
                "safe_download_files": ["config.json", "preprocessor_config.json"],
                "can_auto_repair": True,
            }
        ],
    }
    downloaded: list[str] = []

    def fake_download(repo_id: str, revision: str | None, filename: str, destination: Path, relative_name: str | None = None) -> Path:
        downloaded.append(filename)
        path = destination / (relative_name or filename)
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr("app.hf_model_downloader._download_file", fake_download)

    execution = execute_missing_file_repair_plan(
        HFModelRef("owner/asr"),
        choice,
        ["model.safetensors", "config.json", "preprocessor_config.json"],
        destination,
        plan,
        allow_downloads=True,
        print_func=lambda text: None,
    )

    assert execution["schema"] == "easy_asr_bench.model_layout_repair_execution.v1"
    assert execution["summary"]["repaired"] == 1
    assert execution["records"][0]["status"] == "repaired"
    assert downloaded == ["config.json", "preprocessor_config.json"]


def test_execute_missing_file_repair_plan_blocks_without_download_permission(tmp_path: Path, monkeypatch):
    destination = tmp_path / "model"
    destination.mkdir()
    choice = DownloadChoice(
        label="Safetensors",
        kind="safetensors",
        primary_files=("model.safetensors",),
        files=("model.safetensors",),
        task_hint="metadata_required",
    )
    plan = {
        "schema": "easy_asr_bench.model_layout_repair_plan.v1",
        "records": [
            {
                "issue_id": "model_layout:incomplete",
                "repair_action": "download_exact_missing_files",
                "safe_download_files": ["config.json"],
                "can_auto_repair": True,
            }
        ],
    }
    monkeypatch.setattr("app.hf_model_downloader._download_file", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected download")))

    execution = execute_missing_file_repair_plan(
        HFModelRef("owner/asr"),
        choice,
        ["model.safetensors", "config.json"],
        destination,
        plan,
        allow_downloads=False,
        print_func=lambda text: None,
    )

    assert execution["summary"]["blocked"] == 1
    assert execution["records"][0]["status"] == "blocked"
    assert "not approved" in execution["records"][0]["block_reason"]


def test_execute_missing_file_repair_plan_blocks_invalid_files(tmp_path: Path):
    destination = tmp_path / "model"
    destination.mkdir()
    choice = DownloadChoice(
        label="Safetensors",
        kind="safetensors",
        primary_files=("model.safetensors",),
        files=("model.safetensors",),
        task_hint="metadata_required",
    )
    plan = {
        "schema": "easy_asr_bench.model_layout_repair_plan.v1",
        "records": [
            {
                "issue_id": "model_layout:bad",
                "repair_action": "download_exact_missing_files",
                "safe_download_files": ["invented.json", "model.safetensors"],
                "can_auto_repair": True,
            }
        ],
    }

    execution = execute_missing_file_repair_plan(
        HFModelRef("owner/asr"),
        choice,
        ["model.safetensors", "config.json"],
        destination,
        plan,
        allow_downloads=True,
        print_func=lambda text: None,
    )

    assert execution["summary"]["blocked"] == 1
    assert execution["records"][0]["status"] == "blocked"
    assert "not safe exact repo additions" in execution["records"][0]["block_reason"]


def test_offer_missing_file_repair_writes_structured_llm_audit_request_when_ambiguous(tmp_path: Path, monkeypatch):
    destination = tmp_path / "model"
    destination.mkdir()
    repo_files = ["model.safetensors", "notes/custom.file"]
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


def test_offer_missing_file_repair_offers_conservative_same_package_sidecars(tmp_path: Path, monkeypatch):
    destination = tmp_path / "model"
    destination.mkdir()
    repo_files = [
        "onnx/encoder.onnx",
        "onnx/encoder.onnx_data",
        "onnx/encoder_fp16.onnx",
        "onnx/encoder_fp16.onnx_data",
        "config.json",
    ]
    choice = DownloadChoice(
        label="ONNX package",
        kind="onnx",
        primary_files=("onnx/encoder.onnx",),
        files=("onnx/encoder.onnx",),
        task_hint="metadata_required",
    )
    downloaded: list[str] = []

    def fake_download(repo_id: str, revision: str | None, filename: str, destination: Path, relative_name: str | None = None) -> Path:
        downloaded.append(filename)
        path = destination / (relative_name or filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return path

    missing_candidate = ModelCandidate(
        candidate_id="incomplete",
        display_name="Incomplete",
        family_name="Incomplete",
        backend="onnx",
        container_format="onnx",
        task="unknown",
        precision="unknown",
        quantization_label="Unknown precision",
        path=destination,
        adapter_name="onnx",
        runnable=False,
        missing_files=["sidecar files"],
    )
    monkeypatch.setattr("app.hf_model_downloader._download_file", fake_download)
    monkeypatch.setattr("app.model_scanner.scan_models", lambda root: ([], [missing_candidate]))

    answers = iter(["y"])
    offer_missing_file_repair(
        HFModelRef("owner/asr"),
        choice,
        repo_files,
        destination,
        input_func=lambda prompt: next(answers),
        print_func=lambda text: None,
    )

    assert "onnx/encoder.onnx_data" in downloaded
    assert "config.json" in downloaded
    assert "onnx/encoder_fp16.onnx" not in downloaded
    assert "onnx/encoder_fp16.onnx_data" not in downloaded


def test_offer_missing_file_repair_imports_validated_llm_recommendation_json(tmp_path: Path, monkeypatch):
    destination = tmp_path / "model"
    destination.mkdir()
    repo_files = ["model.safetensors", "special_processor.json", "tokenizer.json"]
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
        missing_files=["custom processor sidecar"],
    )
    recommendation = json.dumps(
        {
            "schema": "easy_asr_bench.hf_missing_file_recommendation.v1",
            "recommended_files": ["special_processor.json"],
            "reason": "same model package metadata",
            "confidence": "medium",
        }
    )
    monkeypatch.setattr("app.hf_model_downloader._download_file", fake_download)
    monkeypatch.setattr("app.model_scanner.scan_models", lambda root: ([], [missing_candidate]))

    answers = iter(["n", "y", recommendation, "y"])
    offer_missing_file_repair(
        HFModelRef("owner/asr"),
        choice,
        repo_files,
        destination,
        input_func=lambda prompt: next(answers),
        print_func=lambda text: None,
    )

    assert downloaded == ["special_processor.json"]


def test_offer_missing_file_repair_rejects_llm_recommendation_with_invented_files(tmp_path: Path, monkeypatch):
    destination = tmp_path / "model"
    destination.mkdir()
    repo_files = ["model.safetensors", "tokenizer.json"]
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
        return destination / (relative_name or filename)

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
        missing_files=["custom metadata"],
    )
    recommendation = json.dumps(
        {
            "schema": "easy_asr_bench.hf_missing_file_recommendation.v1",
            "recommended_files": ["invented.json"],
            "reason": "guess",
            "confidence": "low",
        }
    )
    monkeypatch.setattr("app.hf_model_downloader._download_file", fake_download)
    monkeypatch.setattr("app.model_scanner.scan_models", lambda root: ([], [missing_candidate]))

    answers = iter(["n", "y", recommendation])
    offer_missing_file_repair(
        HFModelRef("owner/asr"),
        choice,
        repo_files,
        destination,
        input_func=lambda prompt: next(answers),
        print_func=lambda text: None,
    )

    assert downloaded == []
