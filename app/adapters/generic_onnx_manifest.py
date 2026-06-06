from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

from .base import ChunkTranscript, ModelCandidate, ModelRunResult
from ..benchmark import process_memory_mb
from ..precision_detector import normalize_precision_label


class GenericOnnxManifestAdapter:
    name = "generic_onnx_manifest"

    def discover(self, models_root: Path) -> list[ModelCandidate]:
        candidates: list[ModelCandidate] = []
        for manifest in models_root.rglob("modelbench.json"):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception as exc:
                candidates.append(self._bad_candidate(manifest, f"Manifest JSON parse failed: {exc}"))
                continue
            missing = validate_manifest(data, manifest.parent)
            raw, bucket = normalize_precision_label(str(data.get("precision", "unknown")))
            runnable = not missing and data.get("task") == "automatic-speech-recognition"
            candidates.append(
                ModelCandidate(
                    candidate_id=f"manifest_onnx__{manifest.parent.name}".lower().replace(" ", "_"),
                    display_name=str(data.get("display_name", manifest.parent.name)),
                    family_name=manifest.parent.name,
                    backend="onnxruntime",
                    container_format="onnx",
                    task=str(data.get("task", "unknown")),
                    precision=raw,
                    quantization_label=bucket,
                    path=manifest.parent,
                    adapter_name=self.name,
                    runnable=runnable,
                    missing_files=missing,
                    warnings=[] if runnable else ["Generic ONNX manifest is incomplete or not ASR."],
                    help_text="Generic ONNX support requires modelbench.json with built-in ctc decoding metadata.",
                    metadata={"manifest": data},
                )
            )
        return candidates

    def _bad_candidate(self, manifest: Path, warning: str) -> ModelCandidate:
        raw, bucket = normalize_precision_label("unknown")
        return ModelCandidate(
            candidate_id=f"manifest_onnx__{manifest.parent.name}".lower().replace(" ", "_"),
            display_name=manifest.parent.name,
            family_name=manifest.parent.name,
            backend="onnxruntime",
            container_format="onnx",
            task="unknown",
            precision=raw,
            quantization_label=bucket,
            path=manifest.parent,
            adapter_name=self.name,
            runnable=False,
            warnings=[warning],
        )

    def required_dependency_groups(self, candidate: ModelCandidate) -> list[str]:
        return ["onnx"]

    def load(self, candidate: ModelCandidate, runtime_config: dict):
        self.candidate = candidate
        self.runtime_config = runtime_config
        from ..onnx_common import choose_providers, make_session

        manifest = candidate.metadata["manifest"]
        model_path = candidate.path / manifest["files"]["model"]
        self.session = make_session(model_path, choose_providers(runtime_config.get("provider", "auto")), int(runtime_config.get("cpu_threads", 0)))
        return self

    def transcribe_chunks(self, chunks: Sequence, chunk_metadata: list[dict]) -> ModelRunResult:
        manifest = self.candidate.metadata["manifest"]
        decoding = manifest.get("decoding", {})
        if decoding.get("type") != "ctc":
            raise RuntimeError("Generic ONNX v1 supports built-in CTC decoding only.")
        blank = int(decoding.get("blank_token_id", 0))
        vocab = decoding.get("vocab", {})
        transcript_chunks: list[ChunkTranscript] = []
        errors: list[str] = []
        inference_seconds = 0.0
        peak_ram = process_memory_mb()
        from ..frontend import input_features
        from ..onnx_common import session_input_names

        input_name = session_input_names(self.session)[0]
        for chunk, metadata in zip(chunks, chunk_metadata):
            started = time.perf_counter()
            try:
                features = input_features(chunk.samples)
                logits = self.session.run(None, {input_name: features})[0]
                ids = greedy_ctc_ids(logits, blank)
                text = decode_ids(ids, vocab)
            except Exception as exc:
                text = f"[ERROR: chunk failed: {exc}]"
                errors.append(f"{metadata['chunk_id']}: {exc}")
            inference_seconds += time.perf_counter() - started
            peak_ram = max(peak_ram, process_memory_mb())
            transcript_chunks.append(ChunkTranscript(str(metadata["chunk_id"]), float(metadata["start_seconds"]), float(metadata["end_seconds"]), text))
        audio_seconds = sum(float(item["end_seconds"]) - float(item["start_seconds"]) for item in chunk_metadata)
        return ModelRunResult(
            self.candidate,
            transcript_chunks,
            {
                "provider": ",".join(self.session.get_providers()),
                "audio_seconds": audio_seconds,
                "chunk_count": len(chunks),
                "inference_seconds": inference_seconds,
                "total_wall_seconds": inference_seconds,
                "tokens_generated": 0,
                "peak_process_memory_mb": peak_ram,
                "audio_seconds_per_wall_second": audio_seconds / max(0.001, inference_seconds),
            },
            errors,
        )

    def unload(self) -> None:
        self.session = None
        self.candidate = None


def validate_manifest(data: dict, root: Path) -> list[str]:
    missing: list[str] = []
    if data.get("schema") != "easy_asr_bench.model_manifest.v1":
        missing.append("schema=easy_asr_bench.model_manifest.v1")
    files = data.get("files", {})
    if not files.get("model") or not (root / files.get("model", "")).exists():
        missing.append("files.model")
    if data.get("decoding", {}).get("type") == "ctc" and "blank_token_id" not in data.get("decoding", {}):
        missing.append("decoding.blank_token_id")
    return missing


def greedy_ctc_ids(logits: np.ndarray, blank: int) -> list[int]:
    import numpy as np

    ids = np.argmax(logits, axis=-1)
    if ids.ndim > 1:
        ids = ids[0]
    output: list[int] = []
    prev = None
    for token in ids.tolist():
        token = int(token)
        if token != prev and token != blank:
            output.append(token)
        prev = token
    return output


def decode_ids(ids: list[int], vocab: dict) -> str:
    if not vocab:
        return " ".join(str(token) for token in ids)
    inv = {int(v): k for k, v in vocab.items()} if all(isinstance(v, int) for v in vocab.values()) else {int(k): v for k, v in vocab.items()}
    return "".join(inv.get(token, "") for token in ids).replace("|", " ").strip()
