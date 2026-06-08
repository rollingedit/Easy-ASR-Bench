from __future__ import annotations

import json
import hashlib
import re
from dataclasses import replace
from pathlib import Path

from .adapters import BUILTIN_ADAPTERS
from .adapters.base import ModelCandidate
from .adapters.gguf_asr_mmproj import find_gguf_asr_pair
from .precision_detector import detect_from_path, detect_safetensors_folder_precision, indexed_safetensor_missing_files


ADAPTER_PRIORITY = {
    "granite_onnx_ar": 100,
    "granite_onnx_nar": 95,
    "hf_whisper_asr": 90,
    "hf_transformers_asr": 80,
    "faster_whisper": 70,
    "whisper_cpp": 65,
    "openai_whisper_pt": 60,
    "generic_onnx_manifest": 50,
    "gguf_asr_mmproj": 40,
    "gguf_llm_reference": 10,
    "none": 0,
}

TEXT_LLM_SIGNALS = {
    "bloom",
    "causal_lm",
    "falcon",
    "gemma",
    "gpt",
    "gptq",
    "llama",
    "mistral",
    "mixtral",
    "phi",
    "qwen",
    "stablelm",
    "text-generation",
    "awq",
    "exl2",
    "exllama",
}

ASR_CONFIG_SIGNALS = {"whisper", "wav2vec2", "hubert", "speech", "ctc", "seamless", "moonshine", "asr"}
TOKENIZER_FILES = {"tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"}


def candidate_root(candidate: ModelCandidate) -> Path:
    return candidate.path.parent.resolve() if candidate.path.is_file() else candidate.path.resolve()


def _candidate_path_suffix(candidate: ModelCandidate, models_root: Path) -> str:
    root = candidate_root(candidate)
    try:
        rel_path = root.relative_to(models_root.resolve())
    except ValueError:
        rel_path = root
    rel = rel_path.as_posix()
    slug = re.sub(r"[^a-z0-9]+", "_", rel.lower()).strip("_")
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
    return f"{slug or 'root'}__{digest}"


def ensure_unique_candidate_ids(candidates: list[ModelCandidate], models_root: Path) -> list[ModelCandidate]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.candidate_id] = counts.get(candidate.candidate_id, 0) + 1
    seen: set[str] = set()
    unique: list[ModelCandidate] = []
    for candidate in candidates:
        candidate_id = candidate.candidate_id
        if counts[candidate_id] > 1:
            base = candidate_id
            candidate_id = f"{base}__{_candidate_path_suffix(candidate, models_root)}"
            counter = 2
            while candidate_id in seen:
                candidate_id = f"{base}__{_candidate_path_suffix(candidate, models_root)}__{counter}"
                counter += 1
            candidate = replace(candidate, candidate_id=candidate_id)
        seen.add(candidate.candidate_id)
        unique.append(candidate)
    return unique


def missing_names(root: Path, names: list[str]) -> list[str]:
    return [name for name in names if not (root / name).exists()]


def any_exists(root: Path, names: set[str]) -> bool:
    return any((root / name).exists() for name in names)


def external_data_missing_for(onnx_file: Path) -> list[Path]:
    missing: list[Path] = []
    data = onnx_file.with_name(onnx_file.name + "_data")
    sidecars = list(onnx_file.parent.glob(onnx_file.name + "_data*"))
    if not data.exists() and not sidecars and any(marker in onnx_file.name.lower() for marker in ["_fp16", "decoder_model_merged", "audio_encoder"]):
        missing.append(data)
    if onnx_file.name.lower().startswith("decoder_model_merged") and "_fp16" in onnx_file.name.lower():
        extra = onnx_file.with_name(onnx_file.name + "_data_1")
        if not extra.exists():
            missing.append(extra)
    return missing


def looks_like_text_llm(config_path: Path) -> bool:
    if not config_path.exists():
        return False
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    text = json.dumps(data).lower()
    architectures = " ".join(data.get("architectures", [])).lower() if isinstance(data.get("architectures"), list) else ""
    model_type = str(data.get("model_type", "")).lower()
    if config_looks_like_asr(data):
        return False
    if any(signal in text or signal in architectures or signal in model_type for signal in TEXT_LLM_SIGNALS):
        return True
    if "causallm" in architectures.replace("_", "") or "lmheadmodel" in architectures.replace("_", ""):
        return True
    return False


def config_looks_like_asr(data: dict) -> bool:
    text = json.dumps(data).lower()
    architectures = " ".join(data.get("architectures", [])).lower() if isinstance(data.get("architectures"), list) else ""
    model_type = str(data.get("model_type", "")).lower()
    return any(signal in text or signal in architectures or signal in model_type for signal in ASR_CONFIG_SIGNALS)


def unsupported_text_llm_candidate(root: Path) -> ModelCandidate:
    raw, bucket = detect_safetensors_folder_precision(root)
    has_config = (root / "config.json").exists()
    has_safetensors = any(root.glob("*.safetensors"))
    has_tokenizer = any((root / name).exists() for name in TOKENIZER_FILES)
    missing = []
    if not has_config:
        missing.append("config.json")
    if not has_safetensors:
        missing.append("*.safetensors weights")
    if not has_tokenizer:
        missing.append("tokenizer files")
    missing.append("GGUF export (.gguf) for local reference LLM loading")
    return ModelCandidate(
        candidate_id=f"unsupported_text_llm__{root.name}".lower(),
        display_name=root.name,
        family_name=root.name,
        backend="transformers",
        container_format="safetensors",
        task="text-generation",
        precision=raw,
        quantization_label=bucket,
        path=root,
        adapter_name="none",
        runnable=False,
        category="unsupported_llm",
        missing_files=missing,
        warnings=["Hugging Face text/non-ASR safetensors were found, but local reference LLM loading is GGUF-only."],
        help_text="Use a .gguf export for local reference/correction, or choose the manual ChatGPT/Claude workflow.",
    )


def unsupported_asr_candidate(
    root: Path,
    *,
    candidate_id: str,
    display_name: str,
    backend: str,
    container_format: str,
    precision: str = "unknown",
    quantization_label: str = "Unknown precision",
    missing_files: list[str] | None = None,
    warning: str,
    help_text: str,
) -> ModelCandidate:
    return ModelCandidate(
        candidate_id=candidate_id.lower().replace(" ", "_"),
        display_name=display_name,
        family_name=display_name,
        backend=backend,
        container_format=container_format,
        task="automatic-speech-recognition",
        precision=precision,
        quantization_label=quantization_label,
        path=root,
        adapter_name="none",
        runnable=False,
        category="recognized_unsupported_asr",
        missing_files=missing_files or [],
        warnings=[warning],
        help_text=help_text,
    )


def recognize_special_package(path: Path, models_root: Path) -> ModelCandidate | None:
    if path.is_file() and path.suffix.lower() == ".nemo":
        raw, bucket = detect_from_path(path)
        return unsupported_asr_candidate(
            path,
            candidate_id=f"nemo__{path.stem}",
            display_name=path.name,
            backend="nemo",
            container_format="nemo",
            precision=raw,
            quantization_label=bucket,
            warning="NeMo ASR archive detected, but NeMo runtime is not packaged in Easy ASR Bench.",
            help_text="Use a Transformers/safetensors export, CTranslate2, ONNX manifest adapter, or add a dedicated NeMo adapter before this can run.",
        )
    if path.is_dir():
        return (
            recognize_fun_asr_package(path)
            or recognize_ort_edge_package(path)
            or recognize_coreml_package(path)
            or recognize_sherpa_onnx_package(path)
            or recognize_asr_gguf_projector_package(path)
            or recognize_qwen_onnx_package(path)
            or recognize_granite_split_onnx_package(path, models_root)
            or recognize_transformersjs_whisper_onnx_package(path, models_root)
        )
    return None


def recognize_fun_asr_package(root: Path) -> ModelCandidate | None:
    names = {item.name.lower() for item in root.iterdir() if item.is_file()}
    signal = {"config.yaml", "am.mvn", "configuration.json"} & names
    if not signal and not any(name.endswith(".mvn") for name in names):
        return None
    token_present = any(name in names for name in {"tokens.txt", "bpe.model", "tokenizer.model", "vocab.txt", "vocab.json"})
    missing = missing_names(root, ["model.pt", "config.yaml", "am.mvn"])
    if not token_present:
        missing.append("tokenizer/BPE file")
    raw, bucket = detect_from_path(root)
    return unsupported_asr_candidate(
        root,
        candidate_id=f"funasr__{root.name}",
        display_name=root.name,
        backend="funasr",
        container_format="funasr",
        precision=raw,
        quantization_label=bucket,
        missing_files=missing,
        warning="FunASR-style ASR package detected, but FunASR runtime is not packaged in Easy ASR Bench.",
        help_text="Keep model.pt, config.yaml, am.mvn, and tokenizer/BPE files together. Add a dedicated FunASR adapter before this can run.",
    )


def recognize_ort_edge_package(root: Path) -> ModelCandidate | None:
    if not any(root.glob("*.ort")):
        return None
    required = ["preprocess.ort", "encode.ort", "uncached_decode.ort", "cached_decode.ort"]
    missing = missing_names(root, required)
    raw, bucket = detect_from_path(root)
    return unsupported_asr_candidate(
        root,
        candidate_id=f"ort_edge__{root.name}",
        display_name=root.name,
        backend="onnxruntime",
        container_format="ort",
        precision=raw,
        quantization_label=bucket,
        missing_files=missing,
        warning="ORT edge ASR package detected, but .ort runtime graphs are not supported by the current adapters.",
        help_text="Keep preprocess.ort, encode.ort, uncached_decode.ort, and cached_decode.ort together. Add a dedicated ORT adapter before this can run.",
    )


def recognize_coreml_package(root: Path) -> ModelCandidate | None:
    if not any(item.is_dir() and item.suffix.lower() == ".mlmodelc" for item in root.iterdir()):
        return None
    required = [
        "AudioEncoder.mlmodelc",
        "MelSpectrogram.mlmodelc",
        "TextDecoder.mlmodelc",
        "TextDecoderContextPrefill.mlmodelc",
        "config.json",
        "generation_config.json",
    ]
    missing = missing_names(root, required)
    raw, bucket = detect_from_path(root)
    return unsupported_asr_candidate(
        root,
        candidate_id=f"coreml__{root.name}",
        display_name=root.name,
        backend="coreml",
        container_format="mlmodelc",
        precision=raw,
        quantization_label=bucket,
        missing_files=missing,
        warning="Core ML / WhisperKit ASR package detected, but this Windows-first app does not run Core ML models.",
        help_text="Use a Windows-supported export such as HF safetensors, faster-whisper, whisper.cpp GGML, or ONNX.",
    )


def recognize_sherpa_onnx_package(root: Path) -> ModelCandidate | None:
    files = [item.name for item in root.iterdir() if item.is_file()]
    lower = {name.lower() for name in files}
    if not any(name.endswith(("-encoder.onnx", "-encoder.int8.onnx", "-decoder.onnx", "-decoder.int8.onnx", "-tokens.txt")) for name in lower):
        return None
    prefixes = []
    for name in files:
        low = name.lower()
        for marker in ["-encoder.int8.onnx", "-encoder.onnx", "-decoder.int8.onnx", "-decoder.onnx", "-tokens.txt"]:
            if low.endswith(marker):
                prefixes.append(name[: -len(marker)])
    prefix = prefixes[0] if prefixes else root.name
    has_encoder = any(name.lower().startswith(prefix.lower() + "-encoder") and name.lower().endswith(".onnx") for name in files)
    has_decoder = any(name.lower().startswith(prefix.lower() + "-decoder") and name.lower().endswith(".onnx") for name in files)
    has_tokens = any(name.lower().startswith(prefix.lower() + "-tokens") for name in files)
    missing = []
    if not has_encoder:
        missing.append(f"{prefix}-encoder.onnx")
    if not has_decoder:
        missing.append(f"{prefix}-decoder.onnx")
    if not has_tokens:
        missing.append(f"{prefix}-tokens.txt")
    raw, bucket = detect_from_path(root)
    return unsupported_asr_candidate(
        root,
        candidate_id=f"sherpa_onnx__{root.name}",
        display_name=root.name,
        backend="sherpa-onnx",
        container_format="onnx",
        precision=raw,
        quantization_label=bucket,
        missing_files=missing,
        warning="sherpa-onnx Whisper package detected, but sherpa-onnx runtime is not packaged in Easy ASR Bench.",
        help_text="Keep matching encoder, decoder, tokens.txt, and any same-prefix .weights files together. Add a sherpa-onnx adapter before this can run.",
    )


def recognize_asr_gguf_projector_package(root: Path) -> ModelCandidate | None:
    pair = find_gguf_asr_pair(root)
    if pair is None:
        return None
    main_model, projector, missing, warnings = pair
    if main_model is not None and projector is not None and not missing:
        return None
    raw, bucket = detect_from_path(main_model if main_model else root)
    return unsupported_asr_candidate(
        root,
        candidate_id=f"asr_gguf_mmproj__{root.name}",
        display_name=root.name,
        backend="llama.cpp",
        container_format="gguf+mmproj",
        precision=raw,
        quantization_label=bucket,
        missing_files=missing,
        warning=warnings[0] if warnings else "Audio/multimodal ASR GGUF package detected, but it is incomplete or ambiguous.",
        help_text="Keep the main .gguf and matching mmproj .gguf together. Complete pairs run through the llama.cpp MTMD runtime path when the native dependency is available.",
    )


def recognize_qwen_onnx_package(root: Path) -> ModelCandidate | None:
    files = {item.name for item in root.iterdir() if item.is_file()}
    signal = {"encoder.onnx", "decoder_init.onnx", "decoder_step.onnx", "decoder_weights.data", "embed_tokens.bin"} & files
    signal |= {"encoder.int4.onnx", "decoder_init.int4.onnx", "decoder_step.int4.onnx", "decoder_weights.int4.data"} & files
    if not signal:
        return None
    int4 = any(name.endswith(".int4.onnx") or ".int4." in name for name in files)
    graph_set = ["encoder.int4.onnx", "decoder_init.int4.onnx", "decoder_step.int4.onnx", "decoder_weights.int4.data"] if int4 else ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx", "decoder_weights.data"]
    required = graph_set + ["embed_tokens.bin", "config.json", "preprocessor_config.json"]
    missing = missing_names(root, required)
    if not any_exists(root, TOKENIZER_FILES | {"vocab.json", "added_tokens.json"}):
        missing.append("tokenizer/vocab files")
    raw, bucket = detect_from_path(root)
    return unsupported_asr_candidate(
        root,
        candidate_id=f"qwen_onnx__{root.name}",
        display_name=root.name,
        backend="onnxruntime",
        container_format="onnx-qwen-asr",
        precision=raw,
        quantization_label=bucket,
        missing_files=missing,
        warning="Qwen-style autoregressive ONNX ASR package detected, but no adapter for this split decode loop exists yet.",
        help_text="Keep encoder, decoder_init, decoder_step, decoder_weights data, embed_tokens.bin, config, preprocessor, and tokenizer files together. Add a Qwen ONNX adapter before this can run.",
    )


def recognize_transformersjs_whisper_onnx_package(root: Path, models_root: Path) -> ModelCandidate | None:
    search_root = root / "onnx" if (root / "onnx").is_dir() else root
    files = {item.name for item in search_root.iterdir() if item.is_file()}
    has_encoder = any(name.startswith("encoder_model") and name.endswith(".onnx") for name in files)
    has_decoder = any(name.startswith("decoder_model") and name.endswith(".onnx") for name in files)
    has_decoder_with_past = any(name.startswith("decoder_with_past_model") and name.endswith(".onnx") for name in files)
    if not (has_encoder or has_decoder or has_decoder_with_past):
        return None
    required_meta = ["config.json", "preprocessor_config.json", "tokenizer.json"]
    missing = missing_names(root, required_meta)
    if not has_encoder:
        missing.append("encoder_model*.onnx")
    if not has_decoder and not has_decoder_with_past:
        missing.append("decoder_model_merged*.onnx or decoder_with_past_model*.onnx")
    for onnx_file in search_root.glob("*.onnx"):
        missing.extend(item.relative_to(root).as_posix() for item in external_data_missing_for(onnx_file))
    raw, bucket = detect_from_path(search_root)
    rel_root = root.relative_to(models_root) if root.is_relative_to(models_root) else root
    return unsupported_asr_candidate(
        root,
        candidate_id=f"whisper_onnx__{root.name}",
        display_name=str(rel_root),
        backend="onnxruntime",
        container_format="onnx-whisper",
        precision=raw,
        quantization_label=bucket,
        missing_files=missing,
        warning="Whisper ONNX encoder/decoder package detected, but the current generic ONNX adapter only runs manifest-described CTC models.",
        help_text="Keep one matching encoder/decoder precision set plus tokenizer/config/preprocessor files. Add a Whisper ONNX adapter or modelbench manifest recipe before this can run.",
    )


def recognize_granite_split_onnx_package(root: Path, models_root: Path) -> ModelCandidate | None:
    search_root = root / "onnx" if (root / "onnx").is_dir() else root
    files = {item.name for item in search_root.iterdir() if item.is_file()}
    has_audio_encoder = any(name.startswith("audio_encoder") and name.endswith(".onnx") for name in files)
    has_decoder = any(name.startswith("decoder_model_merged") and name.endswith(".onnx") for name in files)
    has_embed_tokens = any(name.startswith("embed_tokens") and name.endswith(".onnx") for name in files)
    if not (has_audio_encoder or has_embed_tokens):
        return None
    missing = missing_names(root, ["config.json", "preprocessor_config.json", "tokenizer.json"])
    if not has_audio_encoder:
        missing.append("audio_encoder*.onnx")
    if not has_decoder:
        missing.append("decoder_model_merged*.onnx")
    if not has_embed_tokens:
        missing.append("embed_tokens*.onnx")
    for onnx_file in search_root.glob("*.onnx"):
        missing.extend(item.relative_to(root).as_posix() for item in external_data_missing_for(onnx_file))
    raw, bucket = detect_from_path(search_root)
    rel_root = root.relative_to(models_root) if root.is_relative_to(models_root) else root
    return unsupported_asr_candidate(
        root,
        candidate_id=f"split_onnx_asr__{root.name}",
        display_name=str(rel_root),
        backend="onnxruntime",
        container_format="onnx-split-asr",
        precision=raw,
        quantization_label=bucket,
        missing_files=missing,
        warning="Split audio-encoder/decoder ONNX ASR package detected, but this layout does not match the current built-in ONNX AR/NAR adapters.",
        help_text="Keep audio_encoder, decoder_model_merged, embed_tokens, all .onnx_data sidecars, config, preprocessor, and tokenizer files together. Add a dedicated split-ONNX adapter before this can run.",
    )


def scan_models(models_root: Path) -> tuple[list[ModelCandidate], list[ModelCandidate]]:
    models_root.mkdir(parents=True, exist_ok=True)
    discovered: list[ModelCandidate] = []
    for adapter in BUILTIN_ADAPTERS:
        discovered.extend(adapter.discover(models_root))
    asr_gguf_roots = {
        path.resolve()
        for path in [models_root, *[item for item in models_root.rglob("*") if item.is_dir()]]
        if find_gguf_asr_pair(path) is not None
    }
    discovered = [
        candidate
        for candidate in discovered
        if not (candidate.adapter_name == "gguf_llm_reference" and candidate.path.parent.resolve() in asr_gguf_roots)
    ]

    known_paths = {candidate.path.resolve() for candidate in discovered}
    known_parent_paths = {candidate_root(candidate) for candidate in discovered}
    known_runnable_parent_paths = {candidate_root(candidate) for candidate in discovered if candidate.runnable}
    unsupported: list[ModelCandidate] = []
    precision_folders = {"int8", "fp16w", "fp32", "f32", "float32"}

    for folder in [path for path in models_root.rglob("*") if path.is_dir() and path.name.lower() in precision_folders]:
        files = {item.name for item in folder.iterdir() if item.is_file()}
        parent = folder.parent
        raw, bucket = detect_from_path(folder)
        if {"encoder.onnx", "prompt_encode.onnx", "decode_step.onnx", "embed_tokens.onnx"} & files:
            required = [
                "encoder.onnx",
                "encoder.onnx_data",
                "prompt_encode.onnx",
                "prompt_encode.onnx_data",
                "decode_step.onnx",
                "decode_step.onnx_data",
                "embed_tokens.onnx",
                "embed_tokens.onnx_data",
            ]
            missing = [f"{folder.name}/{name}" for name in required if not (folder / name).exists()]
            for shared in ["tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"]:
                if not (parent / shared).exists():
                    missing.append(shared)
            if missing and parent.resolve() not in known_parent_paths:
                unsupported.append(
                    ModelCandidate(
                        candidate_id=f"incomplete_granite_ar__{parent.name}__{folder.name}".lower(),
                        display_name=f"{parent.name} / {folder.name}",
                        family_name=parent.name,
                        backend="onnxruntime",
                        container_format="onnx",
                        task="automatic-speech-recognition",
                        precision=raw,
                        quantization_label=bucket,
                        path=parent,
                        adapter_name="granite_onnx_ar",
                        runnable=False,
                        missing_files=missing,
                        warnings=["Multi-file ONNX AR-like folder is incomplete."],
                    )
                )
        elif {"encoder.onnx", "editor.onnx", "embed_tokens.onnx"} & files:
            required = [
                "encoder.onnx",
                "encoder.onnx_data",
                "editor.onnx",
                "editor.onnx_data",
                "embed_tokens.onnx",
                "embed_tokens.onnx_data",
            ]
            missing = [f"{folder.name}/{name}" for name in required if not (folder / name).exists()]
            for shared in ["tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"]:
                if not (parent / shared).exists():
                    missing.append(shared)
            if missing and parent.resolve() not in known_parent_paths:
                unsupported.append(
                    ModelCandidate(
                        candidate_id=f"incomplete_granite_nar__{parent.name}__{folder.name}".lower(),
                        display_name=f"{parent.name} / {folder.name}",
                        family_name=parent.name,
                        backend="onnxruntime",
                        container_format="onnx",
                        task="automatic-speech-recognition",
                        precision=raw,
                        quantization_label=bucket,
                        path=parent,
                        adapter_name="granite_onnx_nar",
                        runnable=False,
                        missing_files=missing,
                        warnings=["Multi-file ONNX NAR-like folder is incomplete."],
                    )
                )

    recognized_package_roots: set[Path] = set()
    root_asr_gguf = recognize_asr_gguf_projector_package(models_root)
    if root_asr_gguf is not None:
        unsupported.append(root_asr_gguf)
    for folder in [path for path in models_root.rglob("*") if path.is_dir()]:
        if any(parent.resolve() in recognized_package_roots for parent in [folder, *folder.parents]):
            continue
        if folder.resolve() in known_runnable_parent_paths or any(parent.resolve() in known_runnable_parent_paths for parent in folder.parents):
            continue
        if folder.resolve() not in known_parent_paths and any(parent.resolve() in known_parent_paths for parent in folder.parents):
            continue
        special = recognize_special_package(folder, models_root)
        if special is not None:
            unsupported.append(special)
            recognized_package_roots.add(folder.resolve())
            continue
        if folder.resolve() in known_parent_paths:
            continue

    for path in models_root.rglob("*"):
        if path.resolve() in known_paths:
            continue
        if any(parent.resolve() in recognized_package_roots for parent in [path.parent, *path.parent.parents]):
            continue
        if path.parent.resolve() in known_runnable_parent_paths or any(parent.resolve() in known_runnable_parent_paths for parent in path.parents):
            continue
        if path.resolve() not in known_parent_paths and any(parent.resolve() in known_parent_paths for parent in path.parents):
            continue
        special = recognize_special_package(path, models_root)
        if special is not None:
            unsupported.append(special)
            recognized_package_roots.add(candidate_root(special))
            continue
        if path.is_file() and path.suffix.lower() == ".safetensors" and looks_like_text_llm(path.parent / "config.json"):
            unsupported.append(unsupported_text_llm_candidate(path.parent))
            continue
        if path.parent.resolve() in known_parent_paths:
            continue
        if any(part.lower() in precision_folders for part in path.parts):
            continue
        if path.suffix.lower() in {".pt", ".bin", ".gguf"} and path.resolve() in known_paths:
            continue
        if path.is_file() and path.suffix.lower() == ".gguff":
            raw, bucket = detect_from_path(path)
            unsupported.append(
                ModelCandidate(
                    candidate_id=f"unsupported__{path.stem}".lower(),
                    display_name=path.name,
                    family_name=path.stem,
                    backend="unknown",
                    container_format="gguff",
                    task="unknown",
                    precision=raw,
                    quantization_label=bucket,
                    path=path,
                    adapter_name="none",
                    runnable=False,
                    warnings=["This looks like a typo for .gguf."],
                )
            )
        elif path.is_file() and path.suffix.lower() == ".gguf":
            continue
        elif path.is_file() and path.suffix.lower() == ".onnx":
            if (path.parent / "modelbench.json").exists():
                continue
            raw, bucket = detect_from_path(path)
            manifest = path.parent / "modelbench.json"
            unsupported.append(
                ModelCandidate(
                    candidate_id=f"onnx__{path.stem}".lower(),
                    display_name=path.name,
                    family_name=path.parent.name,
                    backend="onnxruntime",
                    container_format="onnx",
                    task="unknown",
                    precision=raw,
                    quantization_label=bucket,
                    path=path,
                    adapter_name="generic_onnx_manifest",
                    runnable=False,
                    missing_files=[] if manifest.exists() else ["modelbench.json"],
                    warnings=["Generic ONNX requires a supported ASR manifest."],
                )
            )
        elif path.is_file() and path.suffix.lower() == ".safetensors":
            if path.parent.resolve() in known_parent_paths:
                continue
            root = path.parent
            raw, bucket = detect_safetensors_folder_precision(root)
            has_config = (root / "config.json").exists()
            if looks_like_text_llm(root / "config.json"):
                unsupported.append(unsupported_text_llm_candidate(root))
                continue
            has_processor = any((root / name).exists() for name in ["preprocessor_config.json", "processor_config.json"])
            missing = []
            if not has_config:
                missing.append("config.json")
            if not has_processor:
                missing.append("preprocessor_config.json or processor_config.json")
            missing.extend(indexed_safetensor_missing_files(root))
            unsupported.append(
                ModelCandidate(
                    candidate_id=f"safetensors__{root.name}".lower(),
                    display_name=root.name,
                    family_name=root.name,
                    backend="transformers",
                    container_format="safetensors",
                    task="unknown",
                    precision=raw,
                    quantization_label=bucket,
                    path=root,
                    adapter_name="hf_transformers_asr",
                    runnable=False,
                    missing_files=missing,
                    warnings=["Safetensors weights were found, but this folder is not a complete runnable ASR model folder."],
                )
            )

    unique_unsupported: dict[str, ModelCandidate] = {}
    for candidate in unsupported:
        unique_unsupported[candidate.candidate_id + "::" + str(candidate.path.resolve())] = candidate
    runnable = dedupe_runnable([candidate for candidate in discovered if candidate.runnable])
    runnable_roots = {candidate_root(candidate) for candidate in runnable}
    partial = [
        candidate
        for candidate in discovered
        if not candidate.runnable and candidate_root(candidate) not in runnable_roots
        and not any(
            parent.resolve() in recognized_package_roots
            for parent in [candidate_root(candidate), *candidate_root(candidate).parents]
        )
    ]
    unsupported_output = partial + list(unique_unsupported.values())
    return ensure_unique_candidate_ids(runnable, models_root), ensure_unique_candidate_ids(unsupported_output, models_root)


def dedupe_runnable(candidates: list[ModelCandidate]) -> list[ModelCandidate]:
    by_root: dict[Path, ModelCandidate] = {}
    for candidate in candidates:
        root = candidate_root(candidate)
        current = by_root.get(root)
        if current is None or ADAPTER_PRIORITY.get(candidate.adapter_name, 0) > ADAPTER_PRIORITY.get(current.adapter_name, 0):
            by_root[root] = candidate
    return list(by_root.values())
