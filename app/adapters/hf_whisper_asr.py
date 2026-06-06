from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

from .base import ChunkTranscript, ModelCandidate, ModelRunResult
from ..benchmark import process_memory_mb
from ..precision_detector import detect_safetensors_folder_precision


class HFWhisperASRAdapter:
    name = "hf_whisper_asr"

    def __init__(self) -> None:
        self.candidate: ModelCandidate | None = None
        self.pipe = None
        self.runtime_config: dict = {}

    def discover(self, models_root: Path) -> list[ModelCandidate]:
        candidates: list[ModelCandidate] = []
        for folder in [models_root, *[p for p in models_root.rglob("*") if p.is_dir()]]:
            config = folder / "config.json"
            if not config.exists() or not list(folder.glob("*.safetensors")):
                continue
            try:
                data = json.loads(config.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(data.get("model_type", "")).lower() != "whisper" and "whisper" not in json.dumps(data).lower():
                continue
            missing = []
            if not any((folder / name).exists() for name in ["preprocessor_config.json", "processor_config.json"]):
                missing.append("preprocessor_config.json or processor_config.json")
            if not any((folder / name).exists() for name in ["tokenizer.json", "tokenizer_config.json"]):
                missing.append("tokenizer files")
            raw, bucket = detect_safetensors_folder_precision(folder)
            candidates.append(
                ModelCandidate(
                    candidate_id=f"hf_whisper__{folder.name}".lower().replace(" ", "_"),
                    display_name=f"{folder.name} (HF Whisper)",
                    family_name=folder.name,
                    backend="transformers",
                    container_format="safetensors",
                    task="automatic-speech-recognition",
                    precision=raw,
                    quantization_label=bucket,
                    path=folder,
                    adapter_name=self.name,
                    runnable=not missing,
                    runnable_after_dependency_install=not missing,
                    missing_files=missing,
                    warnings=[] if not missing else ["HF Whisper folder is incomplete."],
                    help_text="Use a complete Hugging Face Whisper Safetensors folder.",
                    metadata={"model_type": "whisper"},
                )
            )
        return candidates

    def required_dependency_groups(self, candidate: ModelCandidate) -> list[str]:
        return ["transformers_cpu"]

    def load(self, candidate: ModelCandidate, runtime_config: dict):
        self.candidate = candidate
        self.runtime_config = runtime_config
        try:
            import torch
            from transformers import pipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError("HF Whisper requires the transformers_cpu dependency group.") from exc
        wants_cuda = runtime_config.get("provider") == "cuda" or bool(runtime_config.get("prefer_gpu", False))
        device = 0 if wants_cuda and torch.cuda.is_available() else -1
        whisper_config = runtime_config.get("whisper", {})
        generate_kwargs = {}
        language = runtime_config.get("language", "auto")
        if language and language != "auto":
            generate_kwargs["language"] = language
        task = runtime_config.get("task", "transcribe")
        if task:
            generate_kwargs["task"] = task
        model_kwargs = {}
        if device >= 0:
            model_kwargs["torch_dtype"] = torch.float16
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=str(candidate.path),
            device=device,
            trust_remote_code=False,
            model_kwargs=model_kwargs or None,
            generate_kwargs=generate_kwargs or None,
        )
        return self

    def transcribe_chunks(self, chunks: Sequence, chunk_metadata: list[dict]) -> ModelRunResult:
        if self.pipe is None or self.candidate is None:
            raise RuntimeError("Adapter is not loaded")
        transcript_chunks: list[ChunkTranscript] = []
        errors: list[str] = []
        inference_seconds = 0.0
        peak_ram = process_memory_mb()
        for chunk, metadata in zip(chunks, chunk_metadata):
            started = time.perf_counter()
            try:
                whisper_config = self.runtime_config.get("whisper", {})
                call_kwargs = {
                    "chunk_length_s": int(whisper_config.get("chunk_length_s", 30)),
                    "stride_length_s": int(whisper_config.get("stride_length_s", 5)),
                    "batch_size": int(whisper_config.get("batch_size", 1)),
                    "return_timestamps": bool(whisper_config.get("return_timestamps", False)),
                }
                result = self.pipe({"array": chunk.samples, "sampling_rate": 16000}, **call_kwargs)
                text = result.get("text", "") if isinstance(result, dict) else str(result)
            except Exception as exc:
                text = f"[ERROR: chunk failed: {exc}]"
                errors.append(f"{metadata['chunk_id']}: {exc}")
            inference_seconds += time.perf_counter() - started
            peak_ram = max(peak_ram, process_memory_mb())
            transcript_chunks.append(ChunkTranscript(str(metadata["chunk_id"]), float(metadata["start_seconds"]), float(metadata["end_seconds"]), text.strip()))
        audio_seconds = sum(float(item["end_seconds"]) - float(item["start_seconds"]) for item in chunk_metadata)
        return ModelRunResult(
            self.candidate,
            transcript_chunks,
            {
                "provider": "transformers",
                "device": "cuda" if self.pipe is not None and getattr(self.pipe, "device", None) is not None and device_is_cuda(self.pipe.device) else "cpu",
                "audio_seconds": audio_seconds,
                "chunk_count": len(chunks),
                "inference_seconds": inference_seconds,
                "total_wall_seconds": inference_seconds,
                "peak_process_memory_mb": peak_ram,
                "audio_seconds_per_wall_second": audio_seconds / max(0.001, inference_seconds),
            },
            errors,
        )

    def unload(self) -> None:
        self.pipe = None
        self.candidate = None


def device_is_cuda(device) -> bool:
    return "cuda" in str(device).lower()
