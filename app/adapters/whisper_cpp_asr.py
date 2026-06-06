from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from .base import ChunkTranscript, ModelCandidate, ModelRunResult
from ..benchmark import process_memory_mb
from ..precision_detector import detect_from_path


class WhisperCppASRAdapter:
    name = "whisper_cpp"

    def __init__(self) -> None:
        self.model = None
        self.candidate: ModelCandidate | None = None

    def discover(self, models_root: Path) -> list[ModelCandidate]:
        candidates: list[ModelCandidate] = []
        for path in models_root.rglob("ggml-*.bin"):
            raw, bucket = detect_from_path(path)
            candidates.append(
                ModelCandidate(
                    candidate_id=f"whisper_cpp__{path.stem}".lower().replace(" ", "_"),
                    display_name=f"{path.name} (whisper.cpp)",
                    family_name=path.stem,
                    backend="whisper.cpp",
                    container_format="ggml-bin",
                    task="automatic-speech-recognition",
                    precision=raw,
                    quantization_label=bucket,
                    path=path,
                    adapter_name=self.name,
                    runnable=True,
                    runnable_after_dependency_install=True,
                    help_text="Whisper.cpp GGML model file.",
                )
            )
        return candidates

    def required_dependency_groups(self, candidate: ModelCandidate) -> list[str]:
        return ["whisper_cpp"]

    def load(self, candidate: ModelCandidate, runtime_config: dict):
        try:
            from pywhispercpp.model import Model
        except ModuleNotFoundError as exc:
            raise RuntimeError("whisper.cpp support requires requirements/whisper_cpp.txt.") from exc
        self.candidate = candidate
        self.model = Model(str(candidate.path))
        return self

    def transcribe_chunks(self, chunks: Sequence, chunk_metadata: list[dict]) -> ModelRunResult:
        out: list[ChunkTranscript] = []
        errors: list[str] = []
        inference_seconds = 0.0
        peak_ram = process_memory_mb()
        for chunk, metadata in zip(chunks, chunk_metadata):
            started = time.perf_counter()
            try:
                segments = self.model.transcribe(chunk.samples)
                text = " ".join(getattr(segment, "text", str(segment)).strip() for segment in segments).strip()
            except Exception as exc:
                text = f"[ERROR: chunk failed: {exc}]"
                errors.append(f"{metadata['chunk_id']}: {exc}")
            inference_seconds += time.perf_counter() - started
            peak_ram = max(peak_ram, process_memory_mb())
            out.append(ChunkTranscript(str(metadata["chunk_id"]), float(metadata["start_seconds"]), float(metadata["end_seconds"]), text))
        audio_seconds = sum(float(item["end_seconds"]) - float(item["start_seconds"]) for item in chunk_metadata)
        return ModelRunResult(self.candidate, out, {"provider": "whisper.cpp", "audio_seconds": audio_seconds, "chunk_count": len(chunks), "inference_seconds": inference_seconds, "total_wall_seconds": inference_seconds, "peak_process_memory_mb": peak_ram, "audio_seconds_per_wall_second": audio_seconds / max(0.001, inference_seconds)}, errors)

    def unload(self) -> None:
        self.model = None
        self.candidate = None
