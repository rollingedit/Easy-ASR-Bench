from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Sequence

from .base import ChunkTranscript, ModelCandidate, ModelRunResult
from ..benchmark import VariantMetrics, process_memory_mb
from ..precision_detector import normalize_precision_label


REQUIRED_NAR_FILES = [
    "encoder.onnx",
    "encoder.onnx_data",
    "editor.onnx",
    "editor.onnx_data",
    "embed_tokens.onnx",
    "embed_tokens.onnx_data",
]


class GraniteOnnxNARAdapter:
    name = "granite_onnx_nar"

    def __init__(self) -> None:
        self.runner: GraniteOnnxNAR | None = None
        self.candidate: ModelCandidate | None = None
        self.runtime_config: dict | None = None

    def discover(self, models_root: Path) -> list[ModelCandidate]:
        candidates: list[ModelCandidate] = []
        roots = [models_root]
        roots.extend(path for path in models_root.rglob("*") if path.is_dir())
        seen: set[Path] = set()
        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            if not (root / "tokenizer.json").exists():
                continue
            precision_dirs = [p for p in ["int8", "fp16w", "fp32", "f32", "float32"] if (root / p).is_dir()]
            for precision in precision_dirs:
                raw_precision, precision_bucket = normalize_precision_label(precision)
                folder = root / precision
                missing = [name for name in REQUIRED_NAR_FILES if not (folder / name).exists()]
                has_nar_signal = (folder / "editor.onnx").exists()
                if not has_nar_signal and missing:
                    continue
                candidates.append(
                    ModelCandidate(
                        candidate_id=f"{root.name}__granite_nar__{precision}".lower().replace(" ", "_"),
                        display_name="Multi-file ONNX NAR",
                        family_name=root.name,
                        backend="onnxruntime",
                        container_format="onnx",
                        task="automatic-speech-recognition",
                        precision=raw_precision,
                        quantization_label=precision_bucket,
                        path=root,
                        adapter_name=self.name,
                        runnable=not missing,
                        missing_files=[f"{precision}/{name}" for name in missing],
                        warnings=[] if not missing else ["Multi-file ONNX NAR folder is incomplete."],
                    )
                )
        return candidates

    def required_dependency_groups(self, candidate: ModelCandidate) -> list[str]:
        return ["onnx"]

    def load(self, candidate: ModelCandidate, runtime_config: dict):
        self.candidate = candidate
        self.runtime_config = runtime_config
        from ..onnx_nar import GraniteOnnxNAR

        self.runner = GraniteOnnxNAR(
            candidate.path,
            candidate.precision,
            provider=runtime_config.get("provider", "auto"),
            cpu_threads=int(runtime_config.get("cpu_threads", 0)),
        )
        return self

    def transcribe_chunks(self, chunks: Sequence, chunk_metadata: list[dict]) -> ModelRunResult:
        if self.runner is None or self.candidate is None:
            raise RuntimeError("Adapter is not loaded")
        audio_seconds = sum(float(item["end_seconds"]) - float(item["start_seconds"]) for item in chunk_metadata)
        started = time.perf_counter()
        transcript_chunks: list[ChunkTranscript] = []
        tokens = 0
        inference_seconds = 0.0
        peak_ram = process_memory_mb()
        for chunk, metadata in zip(chunks, chunk_metadata):
            try:
                result = self.runner.transcribe_array(chunk.samples)
                text = str(result.get("text", ""))
                tokens += int(result.get("tokens_generated", 0))
                inference_seconds += float(result.get("inference_seconds", 0.0))
            except Exception as exc:
                result = {"error": str(exc)}
                text = f"[ERROR: chunk failed: {exc}]"
            peak_ram = max(peak_ram, process_memory_mb())
            transcript_chunks.append(
                ChunkTranscript(
                    chunk_id=str(metadata["chunk_id"]),
                    start_seconds=float(metadata["start_seconds"]),
                    end_seconds=float(metadata["end_seconds"]),
                    text=text,
                    raw=result,
                )
            )
        metrics = VariantMetrics(
            variant=self.candidate.candidate_id,
            precision=self.candidate.precision,
            provider=self.runner.actual_provider,
            audio_seconds=audio_seconds,
            chunk_count=len(chunks),
            model_load_seconds=self.runner.model_load_seconds,
            inference_seconds=inference_seconds,
            total_wall_seconds=time.perf_counter() - started,
            tokens_generated=tokens,
            peak_process_memory_mb=peak_ram,
        )
        return ModelRunResult(self.candidate, transcript_chunks, metrics.__dict__, [])

    def unload(self) -> None:
        self.runner = None
        self.candidate = None
        gc.collect()
