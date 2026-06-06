from __future__ import annotations

import json
from pathlib import Path

from .adapters import BUILTIN_ADAPTERS
from .adapters.base import ModelCandidate
from .precision_detector import detect_from_path, detect_safetensors_folder_precision


ADAPTER_PRIORITY = {
    "granite_onnx_ar": 100,
    "granite_onnx_nar": 95,
    "hf_whisper_asr": 90,
    "hf_transformers_asr": 80,
    "faster_whisper": 70,
    "whisper_cpp": 65,
    "openai_whisper_pt": 60,
    "generic_onnx_manifest": 50,
    "gguf_llm_reference": 10,
    "none": 0,
}

TEXT_LLM_SIGNALS = {
    "bloom",
    "causal_lm",
    "falcon",
    "gemma",
    "gpt",
    "llama",
    "mistral",
    "mixtral",
    "phi",
    "qwen",
    "stablelm",
    "text-generation",
}

ASR_CONFIG_SIGNALS = {"whisper", "wav2vec2", "hubert", "speech", "ctc", "seamless", "moonshine", "asr"}
TOKENIZER_FILES = {"tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"}


def candidate_root(candidate: ModelCandidate) -> Path:
    return candidate.path.parent.resolve() if candidate.path.is_file() else candidate.path.resolve()


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
    structural_keys = {"vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads"}
    return len(structural_keys & set(data)) >= 3


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


def scan_models(models_root: Path) -> tuple[list[ModelCandidate], list[ModelCandidate]]:
    models_root.mkdir(parents=True, exist_ok=True)
    discovered: list[ModelCandidate] = []
    for adapter in BUILTIN_ADAPTERS:
        discovered.extend(adapter.discover(models_root))

    known_paths = {candidate.path.resolve() for candidate in discovered}
    known_parent_paths = {candidate_root(candidate) for candidate in discovered}
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

    for path in models_root.rglob("*"):
        if path.resolve() in known_paths:
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
            has_safetensors = any(root.glob("*.safetensors"))
            has_tokenizer = any((root / name).exists() for name in TOKENIZER_FILES)
            if looks_like_text_llm(root / "config.json"):
                unsupported.append(unsupported_text_llm_candidate(root))
                continue
            has_processor = any((root / name).exists() for name in ["preprocessor_config.json", "processor_config.json"])
            missing = []
            if not has_config:
                missing.append("config.json")
            if not has_processor:
                missing.append("preprocessor_config.json or processor_config.json")
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
    ]
    return runnable, partial + list(unique_unsupported.values())


def dedupe_runnable(candidates: list[ModelCandidate]) -> list[ModelCandidate]:
    by_root: dict[Path, ModelCandidate] = {}
    for candidate in candidates:
        root = candidate_root(candidate)
        current = by_root.get(root)
        if current is None or ADAPTER_PRIORITY.get(candidate.adapter_name, 0) > ADAPTER_PRIORITY.get(current.adapter_name, 0):
            by_root[root] = candidate
    return list(by_root.values())
