from __future__ import annotations

import json
from pathlib import Path

from app.model_scanner import scan_models
from qa.runtime_matrix.common import write_row


def _standalone(root: Path) -> list[Path]:
    path = root / "model.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return [path]


def _text_llm(root: Path) -> list[Path]:
    folder = root / "llm-safetensors"
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        "config.json": json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"], "torch_dtype": "float16"}),
        "tokenizer.json": "{}",
    }
    for name, content in files.items():
        (folder / name).write_text(content, encoding="utf-8")
    weight = folder / "model.safetensors"
    weight.write_bytes(b"")
    return [folder / "config.json", folder / "tokenizer.json", weight]


def _sharded_missing(root: Path) -> list[Path]:
    folder = root / "wav2vec2-sharded"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForCTC"]}), encoding="utf-8")
    (folder / "tokenizer.json").write_text("{}", encoding="utf-8")
    present = folder / "model-00001-of-00002.safetensors"
    present.write_bytes(b"")
    index = folder / "model.safetensors.index.fp32.json"
    index.write_text(json.dumps({"weight_map": {"a": present.name, "b": "model-00002-of-00002.safetensors"}}), encoding="utf-8")
    return [folder / "config.json", folder / "tokenizer.json", present, index]


def _candidate_summary(candidate) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "display_name": candidate.display_name,
        "adapter_name": candidate.adapter_name,
        "container_format": candidate.container_format,
        "category": candidate.category,
        "task": candidate.task,
        "runnable": candidate.runnable,
        "missing_files": list(candidate.missing_files),
        "warnings": list(candidate.warnings),
        "help_text": candidate.help_text,
    }


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    root = evidence_dir / "Models"
    if row_id == "standalone_safetensors_incomplete":
        artifacts = _standalone(root)
        runnable, unsupported = scan_models(root)
        candidate = next((item for item in unsupported if item.container_format == "safetensors"), None)
        passed = not runnable and candidate is not None and "config.json" in candidate.missing_files
        return write_row(
            row_id,
            "pass" if passed else "fail",
            evidence_dir,
            summary="Standalone Safetensors file is classified as incomplete and not runnable." if passed else "Standalone Safetensors classification was incorrect.",
            details={"runnable_count": len(runnable), "unsupported": [_candidate_summary(item) for item in unsupported]},
            artifacts=artifacts,
        )
    if row_id == "hf_text_llm_safetensors_unsupported":
        artifacts = _text_llm(root)
        runnable, unsupported = scan_models(root)
        candidate = next((item for item in unsupported if item.category == "unsupported_llm"), None)
        passed = not runnable and candidate is not None and "GGUF export (.gguf) for local reference LLM loading" in candidate.missing_files
        return write_row(
            row_id,
            "pass" if passed else "fail",
            evidence_dir,
            summary="HF text LLM Safetensors folder is classified as unsupported local LLM and routed to GGUF guidance." if passed else "HF text LLM Safetensors classification was incorrect.",
            details={"runnable_count": len(runnable), "unsupported": [_candidate_summary(item) for item in unsupported]},
            artifacts=artifacts,
        )
    if row_id == "sharded_safetensors_index":
        artifacts = _sharded_missing(root)
        runnable, unsupported = scan_models(root)
        candidate = next((item for item in unsupported if item.adapter_name == "hf_transformers_asr"), None)
        passed = not runnable and candidate is not None and "model-00002-of-00002.safetensors" in candidate.missing_files
        return write_row(
            row_id,
            "pass" if passed else "fail",
            evidence_dir,
            summary="Sharded Safetensors index reports the missing shard and stays non-runnable." if passed else "Sharded Safetensors missing-shard classification was incorrect.",
            details={"runnable_count": len(runnable), "unsupported": [_candidate_summary(item) for item in unsupported]},
            artifacts=artifacts,
        )
    return write_row(row_id, "fail", evidence_dir, summary=f"Unhandled Safetensors classification row: {row_id}")
