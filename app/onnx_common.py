from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def precision_for_variant(variant: str) -> str:
    return "fp16w" if variant.endswith("fp16w") else "int8"


def model_family_for_variant(variant: str) -> str:
    return "nar" if variant.startswith("nar_") else "ar"


def choose_providers(provider: str) -> list[str]:
    available = set(ort.get_available_providers())
    if provider == "cuda":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("CUDA was requested but ONNX Runtime CUDAExecutionProvider is not available. Falling back to CPU.")
        return ["CPUExecutionProvider"]
    if provider == "auto" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def make_session(path: Path, providers: list[str], cpu_threads: int = 0) -> ort.InferenceSession:
    options = ort.SessionOptions()
    if cpu_threads:
        options.intra_op_num_threads = int(cpu_threads)
        options.inter_op_num_threads = int(cpu_threads)
    return ort.InferenceSession(str(path), sess_options=options, providers=providers)


def session_input_names(session: ort.InferenceSession) -> list[str]:
    return [item.name for item in session.get_inputs()]


def session_output_names(session: ort.InferenceSession) -> list[str]:
    return [item.name for item in session.get_outputs()]


def causal_mask(size: int) -> np.ndarray:
    mask = np.zeros((1, 1, size, size), dtype=np.float32)
    rows, cols = np.triu_indices(size, k=1)
    mask[:, :, rows, cols] = np.finfo(np.float32).min
    return mask


def decode_mask(length: int) -> np.ndarray:
    return np.zeros((1, 1, 1, length), dtype=np.float32)


def load_tokenizer(root: Path) -> Tokenizer:
    return Tokenizer.from_file(str(root / "tokenizer.json"))
