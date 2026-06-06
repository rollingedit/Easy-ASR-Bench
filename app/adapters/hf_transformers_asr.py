from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

from .base import ChunkTranscript, ModelCandidate, ModelRunResult
from ..benchmark import process_memory_mb
from ..precision_detector import detect_safetensors_folder_precision


class HFTransformersASRAdapter:
    name = "hf_transformers_asr"

    def __init__(self) -> None:
        self.candidate: ModelCandidate | None = None
        self.pipe = None
        self.runtime_config: dict = {}

    def discover(self, models_root: Path) -> list[ModelCandidate]:
        candidates: list[ModelCandidate] = []
        for folder in [models_root, *[p for p in models_root.rglob("*") if p.is_dir()]]:
            safetensors = list(folder.glob("*.safetensors"))
            if not safetensors:
                continue
            config = folder / "config.json"
            processor_files = [
                folder / "preprocessor_config.json",
                folder / "processor_config.json",
                folder / "tokenizer.json",
                folder / "tokenizer_config.json",
            ]
            missing = []
            if not config.exists():
                missing.append("config.json")
            if not any(path.exists() for path in processor_files):
                missing.append("processor/tokenizer files")
            task_ok, warning = self._looks_like_asr(config)
            raw, bucket = detect_safetensors_folder_precision(folder)
            runnable = config.exists() and not missing and task_ok
            candidates.append(
                ModelCandidate(
                    candidate_id=f"hf_asr__{folder.name}".lower().replace(" ", "_"),
                    display_name=folder.name,
                    family_name=folder.name,
                    backend="transformers",
                    container_format="safetensors",
                    task="automatic-speech-recognition" if task_ok else "unknown",
                    precision=raw,
                    quantization_label=bucket,
                    path=folder,
                    adapter_name=self.name,
                    runnable=runnable,
                    runnable_after_dependency_install=runnable,
                    missing_files=missing,
                    warnings=[] if runnable else [warning or "Safetensors folder is missing ASR metadata."],
                    help_text="Use a complete Hugging Face ASR model folder with config, tokenizer/processor files, and safetensors weights.",
                )
            )
        return candidates

    def _looks_like_asr(self, config_path: Path) -> tuple[bool, str]:
        if not config_path.exists():
            return False, "A standalone safetensors file is weights only; put the complete Hugging Face ASR folder in Models."
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return False, "config.json could not be parsed."
        text = json.dumps(data).lower()
        signals = ["whisper", "wav2vec2", "hubert", "speech", "ctc", "seamless", "moonshine", "asr"]
        if any(signal in text for signal in signals):
            return True, ""
        architectures = " ".join(data.get("architectures", [])).lower() if isinstance(data.get("architectures"), list) else ""
        if "ctc" in architectures or "speech" in architectures or "whisper" in architectures:
            return True, ""
        return False, "config.json does not identify a supported ASR architecture."

    def required_dependency_groups(self, candidate: ModelCandidate) -> list[str]:
        return ["transformers_cpu"]

    def load(self, candidate: ModelCandidate, runtime_config: dict):
        self.candidate = candidate
        self.runtime_config = runtime_config
        try:
            import torch
            from transformers import pipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError("Transformers ASR support requires the transformers_cpu dependency group. Run setup.bat repair or install requirements/transformers_cpu.txt.") from exc
        device = 0 if runtime_config.get("provider") == "cuda" and torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=str(candidate.path),
            device=device,
            trust_remote_code=False,
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
                result = self.pipe({"array": chunk.samples, "sampling_rate": 16000})
                text = result.get("text", "") if isinstance(result, dict) else str(result)
            except Exception as exc:
                text = f"[ERROR: chunk failed: {exc}]"
                errors.append(f"{metadata['chunk_id']}: {exc}")
            inference_seconds += time.perf_counter() - started
            peak_ram = max(peak_ram, process_memory_mb())
            transcript_chunks.append(
                ChunkTranscript(
                    chunk_id=str(metadata["chunk_id"]),
                    start_seconds=float(metadata["start_seconds"]),
                    end_seconds=float(metadata["end_seconds"]),
                    text=text.strip(),
                )
            )
        audio_seconds = sum(float(item["end_seconds"]) - float(item["start_seconds"]) for item in chunk_metadata)
        return ModelRunResult(
            candidate=self.candidate,
            transcript_chunks=transcript_chunks,
            metrics={
                "provider": "transformers",
                "audio_seconds": audio_seconds,
                "chunk_count": len(chunks),
                "inference_seconds": inference_seconds,
                "total_wall_seconds": inference_seconds,
                "tokens_generated": 0,
                "peak_process_memory_mb": peak_ram,
                "audio_seconds_per_wall_second": audio_seconds / max(0.001, inference_seconds),
            },
            errors=errors,
        )

    def unload(self) -> None:
        self.pipe = None
        self.candidate = None
