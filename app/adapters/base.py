from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence


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
    errors: list[str] = field(default_factory=list)


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
