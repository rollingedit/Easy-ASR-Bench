from __future__ import annotations

from pathlib import Path

from .adapters import BUILTIN_ADAPTERS
from .adapters.base import ModelCandidate
from .precision_detector import detect_from_path, detect_safetensors_folder_precision


def scan_models(models_root: Path) -> tuple[list[ModelCandidate], list[ModelCandidate]]:
    models_root.mkdir(parents=True, exist_ok=True)
    discovered: list[ModelCandidate] = []
    for adapter in BUILTIN_ADAPTERS:
        discovered.extend(adapter.discover(models_root))

    known_paths = {candidate.path.resolve() for candidate in discovered}
    unsupported: list[ModelCandidate] = []
    precision_folders = {"int8", "fp16w", "fp32"}

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
            if missing:
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
                        warnings=["Granite AR-like ONNX folder is incomplete."],
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
            if missing:
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
                        warnings=["Granite NAR-like ONNX folder is incomplete."],
                    )
                )

    for path in models_root.rglob("*"):
        if path.resolve() in known_paths:
            continue
        if any(part.lower() in precision_folders for part in path.parts):
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
            raw, bucket = detect_from_path(path)
            unsupported.append(
                ModelCandidate(
                    candidate_id=f"gguf__{path.stem}".lower(),
                    display_name=path.name,
                    family_name=path.stem,
                    backend="llama.cpp-compatible",
                    container_format="gguf",
                    task="text-generation",
                    precision=raw,
                    quantization_label=bucket,
                    path=path,
                    adapter_name="gguf_llm_reference",
                    runnable=False,
                    warnings=["GGUF was recognized as a text LLM candidate, not direct ASR."],
                )
            )
        elif path.is_file() and path.suffix.lower() == ".onnx":
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
            root = path.parent
            raw, bucket = detect_safetensors_folder_precision(root)
            has_config = (root / "config.json").exists()
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
                    warnings=["Safetensors weights were found, but this build does not yet include the HF ASR adapter."],
                )
            )

    unique_unsupported: dict[str, ModelCandidate] = {}
    for candidate in unsupported:
        unique_unsupported[candidate.candidate_id + "::" + str(candidate.path.resolve())] = candidate
    runnable = [candidate for candidate in discovered if candidate.runnable]
    partial = [candidate for candidate in discovered if not candidate.runnable]
    return runnable, partial + list(unique_unsupported.values())
