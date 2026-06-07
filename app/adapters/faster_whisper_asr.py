from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from .base import ChunkTranscript, ModelCandidate, ModelRunResult
from ..benchmark import process_memory_mb
from ..precision_detector import normalize_precision_label
from ..runtime_plan import resolve_runtime_plan


class FasterWhisperASRAdapter:
    name = "faster_whisper"

    def __init__(self) -> None:
        self.model = None
        self.candidate: ModelCandidate | None = None
        self.device = "cpu"
        self.requested_compute_type = "default"
        self.effective_compute_type = "default"
        self.load_warnings: list[str] = []

    def discover(self, models_root: Path) -> list[ModelCandidate]:
        candidates: list[ModelCandidate] = []
        for folder in [p for p in models_root.rglob("*") if p.is_dir()]:
            if not (folder / "model.bin").exists() or not (folder / "config.json").exists():
                continue
            if not any((folder / name).exists() for name in ["tokenizer.json", "vocabulary.json", "vocabulary.txt", "vocab.json"]):
                continue
            label = "unknown"
            name_lower = folder.name.lower()
            for value in ["int8_float16", "int8", "float16", "fp16", "f16", "float32", "fp32", "f32"]:
                if value in name_lower:
                    label = value
                    break
            raw, bucket = normalize_precision_label(label)
            candidates.append(
                ModelCandidate(
                    candidate_id=f"faster_whisper__{folder.name}".lower().replace(" ", "_"),
                    display_name=f"{folder.name} (faster-whisper)",
                    family_name=folder.name,
                    backend="faster-whisper",
                    container_format="ctranslate2",
                    task="automatic-speech-recognition",
                    precision=raw,
                    quantization_label=bucket,
                    path=folder,
                    adapter_name=self.name,
                    runnable=True,
                    runnable_after_dependency_install=True,
                    help_text="CTranslate2/faster-whisper model folder.",
                )
            )
        return candidates

    def required_dependency_groups(self, candidate: ModelCandidate) -> list[str]:
        return ["faster_whisper"]

    def load(self, candidate: ModelCandidate, runtime_config: dict):
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("faster-whisper support requires requirements/faster_whisper.txt.") from exc
        self.candidate = candidate
        plan = resolve_runtime_plan("faster_whisper", runtime_config)
        device = "cuda" if plan.actual_provider == "cuda" and plan.backend_verified else "cpu"
        compute_aliases = {"fp16": "float16", "f16": "float16", "fp32": "float32", "f32": "float32"}
        compute_type = compute_aliases.get(candidate.precision.lower(), candidate.precision)
        compute_type = compute_type if compute_type in {"int8", "int8_float16", "float16", "float32"} else "default"
        effective_compute_type = compute_type
        warnings = []
        if device == "cpu" and compute_type in {"float16", "int8_float16"}:
            effective_compute_type = "float32" if compute_type == "float16" else "int8"
            warnings.append(f"Requested {compute_type} on CPU; CTranslate2 uses {effective_compute_type} effectively.")
        self.device = device
        self.requested_compute_type = compute_type
        self.effective_compute_type = effective_compute_type
        if plan.fallback_reason:
            warnings.append(plan.fallback_reason)
        self.runtime_plan = plan
        self.load_warnings = warnings
        try:
            self.model = WhisperModel(str(candidate.path), device=device, compute_type=effective_compute_type)
        except Exception as exc:
            if device == "cuda" and plan.fallback_allowed:
                fallback_compute = "int8" if effective_compute_type in {"float16", "int8_float16"} else effective_compute_type
                self.load_warnings.append(f"CUDA load failed; retried CPU: {exc}")
                self.device = "cpu"
                self.effective_compute_type = fallback_compute
                self.runtime_plan = plan.__class__(
                    **{
                        **plan.__dict__,
                        "actual_provider": "cpu",
                        "device": "cpu",
                        "backend_verified": False,
                        "fallback_reason": f"CUDA load failed; retried CPU: {exc}",
                    }
                )
                self.model = WhisperModel(str(candidate.path), device="cpu", compute_type=fallback_compute)
            else:
                raise
        return self

    def transcribe_chunks(self, chunks: Sequence, chunk_metadata: list[dict]) -> ModelRunResult:
        errors: list[str] = []
        out: list[ChunkTranscript] = []
        inference_seconds = 0.0
        peak_ram = process_memory_mb()
        for chunk, metadata in zip(chunks, chunk_metadata):
            started = time.perf_counter()
            try:
                segments, info = self.model.transcribe(chunk.samples, beam_size=1)
                text = " ".join(segment.text.strip() for segment in segments).strip()
                raw = {"language": getattr(info, "language", None), "language_probability": getattr(info, "language_probability", None)}
            except Exception as exc:
                text = f"[ERROR: chunk failed: {exc}]"
                raw = {"error": str(exc)}
                errors.append(f"{metadata['chunk_id']}: {exc}")
            inference_seconds += time.perf_counter() - started
            peak_ram = max(peak_ram, process_memory_mb())
            out.append(ChunkTranscript(str(metadata["chunk_id"]), float(metadata["start_seconds"]), float(metadata["end_seconds"]), text, raw))
        audio_seconds = sum(float(item["end_seconds"]) - float(item["start_seconds"]) for item in chunk_metadata)
        metrics = {
            "provider": "faster-whisper",
            "device": self.device,
            "requested_provider": getattr(self, "runtime_plan", None).requested_provider if getattr(self, "runtime_plan", None) else "unknown",
            "actual_provider": getattr(self, "runtime_plan", None).actual_provider if getattr(self, "runtime_plan", None) else self.device,
            "backend_verified": getattr(self, "runtime_plan", None).backend_verified if getattr(self, "runtime_plan", None) else False,
            "requested_compute_type": self.requested_compute_type,
            "effective_compute_type": self.effective_compute_type,
            "audio_seconds": audio_seconds,
            "chunk_count": len(chunks),
            "inference_seconds": inference_seconds,
            "total_wall_seconds": inference_seconds,
            "peak_process_memory_mb": peak_ram,
            "audio_seconds_per_wall_second": audio_seconds / max(0.001, inference_seconds),
        }
        if getattr(self, "load_warnings", []):
            metrics["warnings"] = list(self.load_warnings)
        return ModelRunResult(self.candidate, out, metrics, errors)

    def unload(self) -> None:
        self.model = None
        self.candidate = None
