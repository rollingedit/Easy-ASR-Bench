from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .console_style import key, prompt_label


METADATA_NAMES = {
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "normalizer.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "tekken.json",
    "tokens.txt",
    "vocab.json",
    "vocab.txt",
    "vocabulary.json",
    "vocabulary.txt",
}

LARGE_CHOICE_FILE_COUNT = 20
SAME_PACKAGE_REPAIR_LIMIT = 12


@dataclass(frozen=True)
class HFModelRef:
    repo_id: str
    revision: str | None = None
    subfolder: str = ""


@dataclass(frozen=True)
class DownloadChoice:
    label: str
    kind: str
    primary_files: tuple[str, ...]
    files: tuple[str, ...]
    task_hint: str = "unknown"
    notes: tuple[str, ...] = ()


def parse_hf_model_ref(raw: str) -> HFModelRef:
    value = raw.strip().strip('"').strip("'")
    if not value:
        raise ValueError("Paste a Hugging Face model URL or repo id.")
    parsed = urlparse(value)
    if parsed.netloc.lower() in {"huggingface.co", "www.huggingface.co"}:
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if parts and parts[0] in {"models", "datasets", "spaces"}:
            parts = parts[1:]
        if len(parts) < 2:
            raise ValueError("Hugging Face model links should look like https://huggingface.co/owner/model.")
        repo_id = "/".join(parts[:2])
        revision = None
        subfolder_parts: list[str] = []
        if len(parts) > 3 and parts[2] in {"tree", "blob", "resolve"}:
            revision = parts[3]
            subfolder_parts = parts[4:]
            if parts[2] in {"blob", "resolve"} and subfolder_parts:
                subfolder_parts = subfolder_parts[:-1]
        return HFModelRef(repo_id=repo_id, revision=revision, subfolder="/".join(subfolder_parts))
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", value):
        raise ValueError("Repo ids should look like owner/model.")
    return HFModelRef(repo_id=value)


def parent_refs(ref: HFModelRef) -> list[HFModelRef]:
    subfolder = ref.subfolder.strip("/")
    refs = [HFModelRef(ref.repo_id, ref.revision, subfolder)]
    while subfolder:
        subfolder = subfolder.rsplit("/", 1)[0] if "/" in subfolder else ""
        refs.append(HFModelRef(ref.repo_id, ref.revision, subfolder))
    return refs


def _under_subfolder(name: str, subfolder: str) -> bool:
    return not subfolder or name == subfolder or name.startswith(subfolder.rstrip("/") + "/")


def _basename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _dirname(name: str) -> str:
    return name.rsplit("/", 1)[0] if "/" in name else ""


def _metadata_files(files: list[str], subfolder: str) -> list[str]:
    folder = subfolder.strip("/")
    metadata_dirs = {""}
    while folder:
        metadata_dirs.add(folder)
        folder = folder.rsplit("/", 1)[0] if "/" in folder else ""
    return sorted(name for name in files if _dirname(name) in metadata_dirs and _basename(name) in METADATA_NAMES)


def _metadata_notes(files: tuple[str, ...]) -> tuple[str, ...]:
    names = {_basename(name).lower() for name in files}
    notes: list[str] = []
    if "config.json" not in names:
        notes.append("No config.json was found in the selected package or parent metadata folders.")
    if not (names & {"tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json", "vocab.txt", "tekken.json", "tokens.txt"}):
        notes.append("No tokenizer/vocab file was found; runtime support may need extra files or a custom adapter.")
    if not (names & {"preprocessor_config.json", "processor_config.json"}):
        notes.append("No processor/preprocessor config was found; ASR runtimes may need extra files.")
    return tuple(notes)


def _known_auxiliary(name: str) -> bool:
    base = _basename(name).lower()
    return base in {".gitattributes", "readme.md", "license", "licenses"} or base.endswith((".md", ".py", ".cs"))


def _sidecars_for(files: list[str], primary: str) -> list[str]:
    folder = _dirname(primary)
    base = _basename(primary)
    prefix = f"{folder}/" if folder else ""
    return sorted(
        name
        for name in files
        if name.startswith(prefix + base + "_data") or name.startswith(prefix + base + ".data")
    )


def _matching_projectors(files: list[str], gguf: str) -> list[str]:
    folder = _dirname(gguf)
    ggufs = [name for name in files if name.lower().endswith(".gguf") and _dirname(name) == folder]
    projectors = [name for name in ggufs if _basename(name).lower().startswith(("mmproj", "mmproj-"))]
    if not projectors:
        return []
    stem = _basename(gguf).lower().replace(".gguf", "")
    exactish = [name for name in projectors if stem in _basename(name).lower()]
    main_models = [name for name in ggufs if name not in projectors]
    return exactish or (projectors if len(projectors) == 1 and len(main_models) == 1 else [])


def _gguf_split_parts(files: list[str], gguf: str) -> list[str]:
    name = _basename(gguf)
    match = re.match(r"(.+)-(\d{5})-of-(\d{5})\.gguf$", name, re.IGNORECASE)
    if not match:
        return [gguf]
    prefix, _, total = match.groups()
    folder = _dirname(gguf)
    return sorted(
        item
        for item in files
        if _dirname(item) == folder and re.fullmatch(rf"{re.escape(prefix)}-\d{{5}}-of-{re.escape(total)}\.gguf", _basename(item), re.IGNORECASE)
    )


def _split_parts(files: list[str], filename: str, suffix: str) -> list[str]:
    name = _basename(filename)
    match = re.match(rf"(.+)-(\d{{5}})-of-(\d{{5}}){re.escape(suffix)}$", name, re.IGNORECASE)
    if not match:
        return [filename]
    prefix, _, total = match.groups()
    folder = _dirname(filename)
    return sorted(
        item
        for item in files
        if _dirname(item) == folder and re.fullmatch(rf"{re.escape(prefix)}-\d{{5}}-of-{re.escape(total)}{re.escape(suffix)}", _basename(item), re.IGNORECASE)
    )


def _gguf_choice_name(gguf: str) -> str:
    name = _basename(gguf)
    match = re.match(r"(.+)-00001-of-\d{5}\.gguf$", name, re.IGNORECASE)
    return f"{match.group(1)} split GGUF" if match else Path(name).stem


def _safetensors_choice_name(weight: str) -> str:
    name = _basename(weight)
    match = re.match(r"(.+)-00001-of-\d{5}\.safetensors$", name, re.IGNORECASE)
    return f"{match.group(1)} split Safetensors" if match else name


def _onnx_variant(filename: str) -> str:
    stem = Path(_basename(filename)).stem.lower()
    for marker in ["_q4f16", "_quantized", "_fp16", "_q4"]:
        if stem.endswith(marker):
            return marker.lstrip("_")
    match = re.search(r"(?:^|[._-])(int[248]|fp16|q4f16|q4|quantized)(?:$|[._-])", stem)
    if match:
        return match.group(1)
    return "default"


def _onnx_companion_variant(filename: str) -> str:
    base = _basename(filename).lower()
    for marker in ["q4f16", "quantized", "fp16", "q4", "int8", "int4"]:
        if re.search(rf"(?:^|[._-]){re.escape(marker)}(?:$|[._-])", base):
            return marker
    return "default"


def _is_shared_onnx_companion(filename: str) -> bool:
    base = _basename(filename).lower()
    return base.endswith(("tokens.txt", ".tokens")) or base in {"tokens.txt", "vocab.json", "tokenizer.json"}


def build_download_choices(files: list[str], ref: HFModelRef) -> list[DownloadChoice]:
    relevant = sorted(name for name in files if _under_subfolder(name, ref.subfolder))
    metadata = _metadata_files(files, ref.subfolder)
    choices: list[DownloadChoice] = []

    ggufs = [name for name in relevant if name.lower().endswith(".gguf")]
    handled_ggufs: set[str] = set()
    for gguf in ggufs:
        if gguf in handled_ggufs:
            continue
        if _basename(gguf).lower().startswith(("mmproj", "mmproj-")):
            continue
        split_parts = _gguf_split_parts(relevant, gguf)
        handled_ggufs.update(split_parts)
        projectors = _matching_projectors(relevant, gguf)
        notes = ()
        asr_audio = bool(projectors) or any(signal in _basename(gguf).lower() for signal in ["asr", "audio", "whisper"])
        if asr_audio and not projectors:
            notes = ("No matching mmproj GGUF was found; audio GGUF packages usually need one.",)
        selected = sorted({*split_parts, *projectors})
        task_hint = "asr_audio" if asr_audio else "reference_llm"
        kind_label = "Audio/ASR GGUF" if asr_audio else "GGUF reference LLM"
        choices.append(
            DownloadChoice(
                label=f"{kind_label}: {_gguf_choice_name(gguf)}" + (f" + {_basename(projectors[0])}" if len(projectors) == 1 else ""),
                kind="gguf",
                primary_files=(gguf,),
                files=tuple(selected),
                task_hint=task_hint,
                notes=notes,
            )
        )

    index_files = [name for name in relevant if ".safetensors.index" in _basename(name).lower() and _basename(name).lower().endswith(".json")]
    for index in index_files:
        selected_files = tuple(sorted({index, *metadata}))
        choices.append(
            DownloadChoice(
                label=f"Sharded Safetensors: {index}",
                kind="safetensors_index",
                primary_files=(index,),
                files=selected_files,
                task_hint="metadata_required",
                notes=("Shard files will be read from the downloaded index and fetched after selection.", *_metadata_notes(selected_files)),
            )
        )
    indexed_dirs = {_dirname(name) for name in index_files}
    safetensors = [
        name
        for name in relevant
        if name.lower().endswith(".safetensors")
        and not (_dirname(name) in indexed_dirs and len(_split_parts(relevant, name, ".safetensors")) > 1)
    ]
    handled_safetensors: set[str] = set()
    for weight in safetensors:
        if weight in handled_safetensors:
            continue
        split_parts = _split_parts(relevant, weight, ".safetensors")
        handled_safetensors.update(split_parts)
        selected_files = tuple(sorted({*split_parts, *metadata}))
        choices.append(
            DownloadChoice(
                label=f"Safetensors: {_safetensors_choice_name(weight)}",
                kind="safetensors",
                primary_files=(weight,),
                files=selected_files,
                task_hint="metadata_required",
                notes=_metadata_notes(selected_files),
            )
        )

    onnx_groups: dict[tuple[str, str], list[str]] = {}
    for name in relevant:
        if name.lower().endswith(".onnx"):
            onnx_groups.setdefault((_dirname(name), _onnx_variant(name)), []).append(name)
    for (folder, variant), onnx_files in sorted(onnx_groups.items()):
        folder_prefix = f"{folder}/" if folder else ""
        package_files = []
        for name in relevant:
            if not name.startswith(folder_prefix) or name.lower().endswith(".onnx"):
                continue
            base = _basename(name).lower()
            if _is_shared_onnx_companion(name):
                package_files.append(name)
            elif base.endswith((".data", ".bin")) and ".onnx_data" not in base and _onnx_companion_variant(name) in {variant, "default"}:
                package_files.append(name)
        package_files.extend(onnx_files)
        for primary in onnx_files:
            package_files.extend(_sidecars_for(relevant, primary))
        selected_files = tuple(sorted({*package_files, *metadata}))
        choices.append(
            DownloadChoice(
                label=f"ONNX package: {folder or '(repo root)'} [{variant}]",
                kind="onnx",
                primary_files=tuple(sorted(onnx_files)),
                files=selected_files,
                task_hint="metadata_required",
                notes=("Downloads ONNX graphs and sidecars in this package folder, not every weight variant in the repo.", *_metadata_notes(selected_files)),
            )
        )

    unique: dict[tuple[str, tuple[str, ...]], DownloadChoice] = {}
    for choice in choices:
        unique[(choice.kind, choice.files)] = choice
    return list(unique.values())


def build_smart_download_choices(files: list[str], ref: HFModelRef) -> tuple[HFModelRef, list[DownloadChoice]]:
    for candidate_ref in parent_refs(ref):
        choices = build_download_choices(files, candidate_ref)
        if choices:
            return candidate_ref, choices
    if ref.subfolder:
        relevant = sorted(name for name in files if _under_subfolder(name, ref.subfolder) and not _known_auxiliary(name))
        if relevant:
            selected_files = tuple(sorted({*relevant, *_metadata_files(files, ref.subfolder)}))
            return ref, [
                DownloadChoice(
                    label=f"Unknown folder package: {ref.subfolder}",
                    kind="folder",
                    primary_files=tuple(relevant[:1]),
                    files=selected_files,
                    task_hint="unknown",
                    notes=(
                        "This folder contains files but does not match a known GGUF, Safetensors, or ONNX package. It will be downloaded for inspection, not treated as runnable automatically.",
                        *_metadata_notes(selected_files),
                    ),
                )
            ]
    return ref, []


def _safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip("/")).strip("_")


def destination_for(models_root: Path, ref: HFModelRef, choice: DownloadChoice | None = None) -> Path:
    name = ref.repo_id.replace("/", "__")
    if choice is not None and choice.primary_files:
        primary = choice.primary_files[0]
        if choice.kind == "onnx":
            suffix = _safe_path_part(_dirname(primary) or ref.subfolder or _basename(primary))
        elif choice.kind == "gguf":
            suffix = _safe_path_part(_gguf_choice_name(primary).replace(" split GGUF", ""))
        else:
            suffix = _safe_path_part(Path(_basename(primary)).stem)
        if suffix:
            name += "__" + suffix
    elif ref.subfolder:
        name += "__" + _safe_path_part(ref.subfolder)
    return models_root / name


def _download_file(repo_id: str, revision: str | None, filename: str, destination: Path, relative_name: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    cached = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )
    )
    target = destination / (relative_name or filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, target)
    return target


def _strip_prefix(filename: str, prefix: str) -> str:
    clean = prefix.strip("/")
    if clean and filename.startswith(clean + "/"):
        return filename[len(clean) + 1 :]
    return filename


def local_relative_name(choice: DownloadChoice, filename: str) -> str:
    if choice.kind in {"gguf", "safetensors", "safetensors_index"} and choice.primary_files:
        return _strip_prefix(filename, _dirname(choice.primary_files[0]))
    return filename


def _expand_index_shards(choice: DownloadChoice, destination: Path) -> list[str]:
    files = set(choice.files)
    for index in choice.primary_files:
        local_index = destination / local_relative_name(choice, index)
        if not local_index.exists():
            continue
        try:
            data = json.loads(local_index.read_text(encoding="utf-8"))
        except Exception:
            continue
        weight_map = data.get("weight_map", {})
        if isinstance(weight_map, dict):
            index_dir = _dirname(index)
            for name in weight_map.values():
                if not isinstance(name, str):
                    continue
                files.add(name if "/" in name or not index_dir else f"{index_dir}/{name}")
    return sorted(files)


def download_choice(ref: HFModelRef, choice: DownloadChoice, destination: Path, print_func=print) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    first_pass = list(choice.files)
    for filename in first_pass:
        target = destination / local_relative_name(choice, filename)
        if target.exists():
            print_func(f"Already exists, skipping {local_relative_name(choice, filename)}")
            continue
        print_func(f"Downloading {filename}")
        downloaded.append(_download_file(ref.repo_id, ref.revision, filename, destination, local_relative_name(choice, filename)))
    if choice.kind == "safetensors_index":
        for filename in _expand_index_shards(choice, destination):
            if filename in first_pass:
                continue
            target = destination / local_relative_name(choice, filename)
            if target.exists():
                print_func(f"Already exists, skipping {local_relative_name(choice, filename)}")
                continue
            print_func(f"Downloading {filename}")
            downloaded.append(_download_file(ref.repo_id, ref.revision, filename, destination, local_relative_name(choice, filename)))
    return downloaded


def _remote_missing_candidates(missing: str, files: list[str], choice: DownloadChoice) -> list[str]:
    if "*" in missing or "/" in missing and missing not in files:
        return []
    prefixes = [""]
    if choice.primary_files:
        primary_dir = _dirname(choice.primary_files[0])
        if primary_dir:
            prefixes.append(primary_dir + "/")
    names = {missing}
    if " or " in missing:
        names.update(part.strip() for part in missing.split(" or ") if part.strip())
    candidates: list[str] = []
    for name in names:
        if "/" in name:
            if name in files:
                candidates.append(name)
            continue
        for prefix in prefixes:
            remote = prefix + name
            if remote in files:
                candidates.append(remote)
    return sorted(set(candidates))


def _safe_repair_relative_files(choice: DownloadChoice, files: list[str]) -> list[str]:
    selected = set(choice.files)
    candidates: set[str] = set()
    primary_dirs = {_dirname(name) for name in choice.primary_files}
    parent_dirs = {""}
    for folder in primary_dirs:
        while folder:
            parent_dirs.add(folder)
            folder = folder.rsplit("/", 1)[0] if "/" in folder else ""
    for name in files:
        if name in selected or _known_auxiliary(name):
            continue
        folder = _dirname(name)
        base = _basename(name)
        if folder in parent_dirs and base in METADATA_NAMES:
            candidates.add(name)
            continue
        if choice.kind == "gguf" and choice.task_hint == "asr_audio":
            for primary in choice.primary_files:
                if name in _matching_projectors(files, primary):
                    candidates.add(name)
            continue
        if choice.kind != "onnx" or folder not in primary_dirs:
            continue
        lower = base.lower()
        if lower.endswith(".onnx"):
            continue
        selected_variants = {_onnx_variant(primary) for primary in choice.primary_files}
        if (
            _is_shared_onnx_companion(name)
            or lower.endswith((".onnx_data", ".data", ".bin"))
            and _onnx_companion_variant(name) in {*selected_variants, "default"}
        ):
            candidates.add(name)
    return sorted(candidates)


def _download_validated_files(
    ref: HFModelRef,
    choice: DownloadChoice,
    filenames: list[str],
    destination: Path,
    print_func=print,
) -> list[Path]:
    downloaded: list[Path] = []
    for filename in filenames:
        target = destination / local_relative_name(choice, filename)
        if target.exists():
            print_func(f"Already exists, skipping {local_relative_name(choice, filename)}")
            continue
        print_func(f"Downloading {filename}")
        downloaded.append(_download_file(ref.repo_id, ref.revision, filename, destination, local_relative_name(choice, filename)))
    return downloaded


def _parse_llm_recommendation(raw: str, repo_files: list[str], already_selected: set[str]) -> tuple[list[str], str | None]:
    text = raw.strip()
    if not text:
        return [], "No recommendation JSON was provided."
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], f"Recommendation JSON could not be parsed: {exc}"
    if data.get("schema") != "easy_asr_bench.hf_missing_file_recommendation.v1":
        return [], "Recommendation JSON used the wrong schema."
    recommended = data.get("recommended_files")
    if not isinstance(recommended, list):
        return [], "Recommendation JSON must include a recommended_files list."
    repo_set = set(repo_files)
    validated: list[str] = []
    invalid: list[str] = []
    for item in recommended:
        if not isinstance(item, str):
            invalid.append(str(item))
            continue
        if item not in repo_set:
            invalid.append(item)
            continue
        if item in already_selected:
            continue
        validated.append(item)
    if invalid:
        return [], "Recommendation included files that are not exact paths in the repo file list: " + ", ".join(invalid)
    return sorted(set(validated)), None


def offer_missing_file_repair(
    ref: HFModelRef,
    choice: DownloadChoice,
    repo_files: list[str],
    destination: Path,
    input_func=input,
    print_func=print,
) -> None:
    from .model_scanner import scan_models

    runnable, unsupported = scan_models(destination)
    if runnable:
        return
    missing: list[str] = []
    for candidate in unsupported:
        if destination.resolve() in [candidate.path.resolve(), *candidate.path.resolve().parents]:
            missing.extend(candidate.missing_files)
    missing = sorted(set(item for item in missing if item))
    if not missing:
        return
    print_func("")
    print_func("Downloaded package still looks incomplete after rescan.")
    for item in missing:
        print_func(f"  missing: {item}")
    repair_files: list[str] = []
    for item in missing:
        repair_files.extend(_remote_missing_candidates(item, repo_files, choice))
    repair_files = sorted(set(repair_files) - set(choice.files))
    if repair_files:
        print_func("")
        print_func("Exact missing-file matches are available in the repo:")
        for index, filename in enumerate(repair_files, 1):
            print_func(f"[{key(str(index))}] {filename}")
        answer = input_func(f"Download these missing files now? [{key('Y')}/{key('n')}] ").strip().lower()
        if answer not in {"n", "no"}:
            _download_validated_files(ref, choice, repair_files, destination, print_func=print_func)
            return
        print_func("Skipped exact missing-file repair download.")

    same_package_files = [
        filename
        for filename in _safe_repair_relative_files(choice, repo_files)
        if filename not in {*choice.files, *repair_files} and not (destination / local_relative_name(choice, filename)).exists()
    ]
    if same_package_files:
        print_func("")
        print_func("Conservative same-package repair files are available:")
        for index, filename in enumerate(same_package_files, 1):
            print_func(f"[{key(str(index))}] {filename}")
        if len(same_package_files) > SAME_PACKAGE_REPAIR_LIMIT:
            print_func(f"This repair set has {len(same_package_files)} file(s). Review before downloading.")
        answer = input_func(f"Download these same-package repair files now? [{key('y')}/{key('N')}] ").strip().lower()
        if answer in {"y", "yes"}:
            _download_validated_files(ref, choice, same_package_files, destination, print_func=print_func)
            return
        print_func("Skipped same-package repair download.")

    print_func("No automatic repair download was attempted for the remaining ambiguous requirements.")
    write_llm_file_audit_request(ref, choice, repo_files, destination, missing)
    print_func(f"Wrote structured LLM/file-audit request to {destination / 'hf_missing_file_request.json'}")
    answer = input_func(f"Paste validated LLM recommendation JSON now? [{key('y')}/{key('N')}] ").strip().lower()
    if answer not in {"y", "yes"}:
        return
    raw_recommendation = input_func(prompt_label("Recommendation JSON or file path> ")).strip()
    if raw_recommendation:
        path_text = raw_recommendation.strip('"').strip("'")
        maybe_path = Path(path_text)
        if maybe_path.exists() and maybe_path.is_file():
            raw_recommendation = maybe_path.read_text(encoding="utf-8")
    recommended_files, error = _parse_llm_recommendation(raw_recommendation, repo_files, set(choice.files))
    if error:
        print_func(error)
        return
    if not recommended_files:
        print_func("LLM recommendation did not include any new exact repo files to download.")
        return
    print_func("")
    print_func("Validated LLM-recommended repo files:")
    for index, filename in enumerate(recommended_files, 1):
        print_func(f"[{key(str(index))}] {filename}")
    answer = input_func(f"Download these recommended files now? [{key('y')}/{key('N')}] ").strip().lower()
    if answer in {"y", "yes"}:
        _download_validated_files(ref, choice, recommended_files, destination, print_func=print_func)


def write_llm_file_audit_request(
    ref: HFModelRef,
    choice: DownloadChoice,
    repo_files: list[str],
    destination: Path,
    missing_files: list[str],
) -> None:
    payload = {
        "schema": "easy_asr_bench.hf_missing_file_request.v1",
        "repo_id": ref.repo_id,
        "revision": ref.revision,
        "selected_choice": {
            "label": choice.label,
            "kind": choice.kind,
            "task_hint": choice.task_hint,
            "primary_files": list(choice.primary_files),
            "downloaded_files": list(choice.files),
        },
        "scanner_missing_files": list(missing_files),
        "repo_files": list(repo_files),
        "required_response_schema": {
            "schema": "easy_asr_bench.hf_missing_file_recommendation.v1",
            "recommended_files": ["exact/path/in/repo.ext"],
            "reason": "short explanation",
            "confidence": "high|medium|low",
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "hf_missing_file_request.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    prompt = (
        "You are helping Easy ASR Bench repair an incomplete Hugging Face model package.\n"
        "Return only JSON matching schema easy_asr_bench.hf_missing_file_recommendation.v1.\n"
        "Recommend only exact filenames that appear in repo_files. Do not invent files.\n"
        "If unsure, return an empty recommended_files list and explain why.\n\n"
        f"{json.dumps(payload, indent=2)}\n"
    )
    (destination / "hf_missing_file_prompt.txt").write_text(prompt, encoding="utf-8")


def list_repo_files(ref: HFModelRef) -> list[str]:
    from huggingface_hub import HfApi

    return list(HfApi().list_repo_files(ref.repo_id, revision=ref.revision))


def download_hf_model_interactive(models_root: Path, input_func=input, print_func=print) -> Path | None:
    raw = input_func(prompt_label("Hugging Face model URL or repo id> ")).strip()
    try:
        ref = parse_hf_model_ref(raw)
        files = list_repo_files(ref)
        ref, choices = build_smart_download_choices(files, ref)
    except Exception as exc:
        print_func(f"Could not inspect that Hugging Face model: {exc}")
        return None
    if not choices:
        print_func("No supported GGUF, Safetensors, or ONNX package choices were found in that repo/subfolder.")
        return None
    print_func("")
    print_func("Available download choices:")
    for index, choice in enumerate(choices, 1):
        print_func(f"[{key(str(index))}] {choice.label}")
        print_func(f"    files: {len(choice.files)} initial file(s)")
        if choice.task_hint == "reference_llm":
            print_func("    use: local LLM reference/correction, not direct ASR")
        elif choice.task_hint == "asr_audio":
            print_func("    use: audio/ASR package")
        elif choice.task_hint == "unknown":
            print_func("    use: unknown package for inspection; not automatically runnable")
        for note in choice.notes:
            print_func(f"    note: {note}")
    if len(choices) == 1:
        selected = choices[0]
    else:
        while True:
            raw_choice = input_func(prompt_label("Download choice> ")).strip()
            if raw_choice.isdigit() and 1 <= int(raw_choice) <= len(choices):
                selected = choices[int(raw_choice) - 1]
                break
            print_func("Choose one download number.")
    if selected.task_hint == "unknown" or len(selected.files) > LARGE_CHOICE_FILE_COUNT:
        print_func("")
        print_func(f"Selected package has {len(selected.files)} file(s).")
        if selected.task_hint == "unknown":
            print_func("This is an unknown package layout and may not be runnable.")
        answer = input_func(f"Download this package now? [{key('Y')}/{key('n')}] ").strip().lower()
        if answer in {"n", "no"}:
            print_func("Download cancelled.")
            return None
    destination = destination_for(models_root, ref, selected)
    try:
        downloaded = download_choice(ref, selected, destination, print_func=print_func)
    except Exception as exc:
        print_func(f"Download failed: {exc}")
        return None
    print_func(f"Downloaded {len(downloaded)} file(s) to {destination}")
    offer_missing_file_repair(ref, selected, files, destination, input_func=input_func, print_func=print_func)
    return destination
