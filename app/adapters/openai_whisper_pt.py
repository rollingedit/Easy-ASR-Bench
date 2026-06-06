from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from .base import ChunkTranscript, ModelCandidate, ModelRunResult
from ..benchmark import process_memory_mb
from ..precision_detector import normalize_precision_label


KNOWN_OFFICIAL_SHA256: dict[str, str] = {}
KNOWN_OFFICIAL_NAMES = {"tiny.pt", "base.pt", "small.pt", "medium.pt", "large.pt", "large-v1.pt", "large-v2.pt", "large-v3.pt"}


class OpenAIWhisperPTAdapter:
    name = "openai_whisper_pt"

    def __init__(self) -> None:
        self.model = None
        self.candidate: ModelCandidate | None = None

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
            import whisper
        except ModuleNotFoundError as exc:
            raise RuntimeError("OpenAI Whisper .pt support requires requirements/openai_whisper.txt.") from exc
        self.candidate = candidate
        self.model = whisper.load_model(str(candidate.path))
        return self

    def transcribe_chunks(self, chunks: Sequence, chunk_metadata: list[dict]) -> ModelRunResult:
        out: list[ChunkTranscript] = []
        errors: list[str] = []
        inference_seconds = 0.0
        peak_ram = process_memory_mb()
        for chunk, metadata in zip(chunks, chunk_metadata):
            started = time.perf_counter()
            try:
                result = self.model.transcribe(chunk.samples, fp16=False)
                text = result.get("text", "")
            except Exception as exc:
                text = f"[ERROR: chunk failed: {exc}]"
                errors.append(f"{metadata['chunk_id']}: {exc}")
            inference_seconds += time.perf_counter() - started
            peak_ram = max(peak_ram, process_memory_mb())
            out.append(ChunkTranscript(str(metadata["chunk_id"]), float(metadata["start_seconds"]), float(metadata["end_seconds"]), text.strip()))
        audio_seconds = sum(float(item["end_seconds"]) - float(item["start_seconds"]) for item in chunk_metadata)
        return ModelRunResult(self.candidate, out, {"provider": "openai-whisper", "audio_seconds": audio_seconds, "chunk_count": len(chunks), "inference_seconds": inference_seconds, "total_wall_seconds": inference_seconds, "peak_process_memory_mb": peak_ram, "audio_seconds_per_wall_second": audio_seconds / max(0.001, inference_seconds)}, errors)

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
