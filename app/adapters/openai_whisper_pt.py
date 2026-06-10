from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from .base import ChunkTranscript, ModelCandidate, ModelRunResult, chunk_failure_error
from ..benchmark import process_memory_mb
from ..precision_detector import normalize_precision_label
from ..runtime_plan import resolve_runtime_plan


KNOWN_OFFICIAL_SHA256: dict[str, str] = {
    "tiny.en.pt": "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03",
    "tiny.pt": "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9",
    "base.en.pt": "25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead",
    "base.pt": "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e",
    "small.en.pt": "f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872",
    "small.pt": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
    "medium.en.pt": "d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f",
    "medium.pt": "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1",
    "large-v1.pt": "e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a",
    "large-v2.pt": "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524",
    "large-v3.pt": "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb",
    "large-v3-turbo.pt": "aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a",
}
KNOWN_OFFICIAL_NAMES = set(KNOWN_OFFICIAL_SHA256) | {"large.pt", "turbo.pt"}


class OpenAIWhisperPTAdapter:
    name = "openai_whisper_pt"

    def __init__(self) -> None:
        self.model = None
        self.candidate: ModelCandidate | None = None
        self.runtime_config: dict = {}
        self.runtime_plan = None

    def discover(self, models_root: Path) -> list[ModelCandidate]:
        candidates: list[ModelCandidate] = []
        for path in models_root.rglob("*.pt"):
            verified = is_verified_official_checkpoint(path)
            known_name = path.name.lower() in KNOWN_OFFICIAL_NAMES
            raw, bucket = normalize_precision_label("unknown")
            warnings = []
            if not verified:
                warnings.append("Local .pt checkpoints use pickle and are blocked unless their SHA256 is allowlisted or unsafe loading is explicitly enabled.")
                if known_name:
                    warnings.append("The filename matches an OpenAI Whisper checkpoint name, but filenames are not trusted.")
            candidates.append(
                ModelCandidate(
                    candidate_id=f"openai_whisper_pt__{path.stem}".lower().replace(" ", "_"),
                    display_name=f"{path.name} (OpenAI Whisper .pt)",
                    family_name=path.stem,
                    backend="openai-whisper",
                    container_format="pt",
                    task="automatic-speech-recognition",
                    precision=raw,
                    quantization_label=bucket,
                    path=path,
                    adapter_name=self.name,
                    runnable=verified,
                    runnable_after_dependency_install=verified,
                    warnings=warnings,
                    help_text="OpenAI Whisper .pt files are runnable only when SHA256-verified or when security.allow_pickle_or_pt_files is explicitly enabled.",
                )
            )
        return candidates

    def required_dependency_groups(self, candidate: ModelCandidate) -> list[str]:
        return ["openai_whisper"]

    def load(self, candidate: ModelCandidate, runtime_config: dict):
        security = runtime_config.get("security", {})
        if not is_verified_official_checkpoint(candidate.path) and not security.get("allow_pickle_or_pt_files", False):
            raise RuntimeError("Blocked .pt checkpoint. Filename is not a safety check; set security.allow_pickle_or_pt_files=true only for files you trust.")
        try:
            import torch
            import whisper
        except ModuleNotFoundError as exc:
            raise RuntimeError("OpenAI Whisper .pt support requires requirements/openai_whisper.txt.") from exc
        self.candidate = candidate
        self.runtime_config = runtime_config
        plan = resolve_runtime_plan("openai_whisper", runtime_config)
        self.runtime_plan = plan
        device = "cuda" if plan.actual_provider == "cuda" else None
        try:
            self.model = whisper.load_model(str(candidate.path), device=device) if device else whisper.load_model(str(candidate.path))
        except Exception as exc:
            if device == "cuda" and plan.fallback_allowed:
                self.model = whisper.load_model(str(candidate.path))
                self.runtime_plan = plan.__class__(
                    plan.model_family,
                    plan.requested_provider,
                    "cpu",
                    "cpu",
                    plan.compute_type,
                    False,
                    plan.fallback_allowed,
                    plan.reason,
                    f"CUDA load failed; retried CPU: {exc}",
                )
            else:
                raise
        return self

    def transcribe_chunks(self, chunks: Sequence, chunk_metadata: list[dict]) -> ModelRunResult:
        out: list[ChunkTranscript] = []
        errors: list = []
        inference_seconds = 0.0
        peak_ram = process_memory_mb()
        for chunk, metadata in zip(chunks, chunk_metadata):
            started = time.perf_counter()
            try:
                use_fp16 = str(getattr(self.model, "device", "")).startswith("cuda")
                result = self.model.transcribe(chunk.samples, fp16=use_fp16)
                text = result.get("text", "")
            except Exception as exc:
                text = ""
                errors.append(chunk_failure_error(self.candidate, metadata, exc))
            inference_seconds += time.perf_counter() - started
            peak_ram = max(peak_ram, process_memory_mb())
            out.append(ChunkTranscript(str(metadata["chunk_id"]), float(metadata["start_seconds"]), float(metadata["end_seconds"]), text.strip()))
        audio_seconds = sum(float(item["end_seconds"]) - float(item["start_seconds"]) for item in chunk_metadata)
        device = str(getattr(self.model, "device", "cpu")) if self.model is not None else "cpu"
        metrics = {
            "provider": "openai-whisper",
            "device": device,
            "provider_summary": {
                "requested_provider": getattr(self.runtime_plan, "requested_provider", "unknown"),
                "actual_provider": getattr(self.runtime_plan, "actual_provider", device),
                "backend_verified": getattr(self.runtime_plan, "backend_verified", False),
                "fallback_allowed": getattr(self.runtime_plan, "fallback_allowed", True),
                "reason": getattr(self.runtime_plan, "reason", ""),
                "fallback_reason": getattr(self.runtime_plan, "fallback_reason", None),
            },
            "audio_seconds": audio_seconds,
            "chunk_count": len(chunks),
            "inference_seconds": inference_seconds,
            "total_wall_seconds": inference_seconds,
            "peak_process_memory_mb": peak_ram,
            "audio_seconds_per_wall_second": audio_seconds / max(0.001, inference_seconds),
            "unsafe_pt_loading_enabled": bool(self.runtime_config.get("security", {}).get("allow_pickle_or_pt_files", False)),
            "checkpoint_sha256_verified": bool(self.candidate and is_verified_official_checkpoint(self.candidate.path)),
        }
        return ModelRunResult(self.candidate, out, metrics, errors)

    def unload(self) -> None:
        self.model = None
        self.candidate = None


def sha256_path(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_verified_official_checkpoint(path: Path) -> bool:
    expected = KNOWN_OFFICIAL_SHA256.get(path.name.lower())
    if not expected:
        return False
    try:
        return sha256_path(path).lower() == expected.lower()
    except OSError:
        return False
