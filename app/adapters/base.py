from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass
class ModelCandidate:
    candidate_id: str
    display_name: str
    family_name: str
    backend: str
    container_format: str
    task: str
    precision: str
    quantization_label: str
    path: Path
    adapter_name: str
    runnable: bool
    category: str = "asr"
    runnable_after_dependency_install: bool = False
    dependency_groups: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    help_text: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkTranscript:
    chunk_id: str
    start_seconds: float
    end_seconds: float
    text: str
    raw: dict = field(default_factory=dict)


@dataclass
class ModelRunResult:
    candidate: ModelCandidate
    transcript_chunks: list[ChunkTranscript]
    metrics: dict
    errors: list[str | dict[str, Any]] = field(default_factory=list)


def chunk_failure_error(candidate: ModelCandidate | None, metadata: dict, exc: Exception) -> dict[str, Any]:
    return {
        "status": "chunk_failed",
        "stage": "chunk_inference",
        "chunk_id": str(metadata.get("chunk_id", "")),
        "start_seconds": float(metadata.get("start_seconds", 0.0)),
        "end_seconds": float(metadata.get("end_seconds", 0.0)),
        "model_id": candidate.candidate_id if candidate else "",
        "model_name": candidate.display_name if candidate else "",
        "adapter_name": candidate.adapter_name if candidate else "",
        "error_type": exc.__class__.__name__,
        "message": str(exc) or exc.__class__.__name__,
    }


class ASRAdapter(Protocol):
    name: str

    def discover(self, models_root: Path) -> list[ModelCandidate]:
        ...

    def required_dependency_groups(self, candidate: ModelCandidate) -> list[str]:
        ...

    def load(self, candidate: ModelCandidate, runtime_config: dict):
        ...

    def transcribe_chunks(self, chunks: Sequence, chunk_metadata: list[dict]) -> ModelRunResult:
        ...

    def unload(self) -> None:
        ...
