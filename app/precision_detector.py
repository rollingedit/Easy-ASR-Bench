from __future__ import annotations

import json
import re
from pathlib import Path


def normalize_precision_label(raw: str | None) -> tuple[str, str]:
    label = (raw or "unknown").strip()
    lower = label.lower()
    if lower in {"int4", "q4", "q4_k_m", "q4_k_s", "q4_0", "q4_1"}:
        return label, "4-bit / Q4"
    if lower in {"q5", "q5_k_m", "q5_k_s", "q5_0", "q5_1"}:
        return label, "5-bit / Q5"
    if lower in {"int8", "q8", "q8_0", "i8"}:
        return label, "8-bit / INT8 / Q8"
    if lower in {"fp16", "f16", "float16", "fp16w"}:
        return label, "16-bit / FP16-family"
    if lower in {"bf16", "bfloat16"}:
        return label, "16-bit / BF16"
    if lower in {"fp32", "f32", "float32"}:
        return label, "32-bit / FP32"
    return label, "Unknown precision"


def detect_from_path(path: Path) -> tuple[str, str]:
    parts = [part.lower() for part in path.parts]
    for label in ["int8", "fp16w", "fp32", "fp16", "bf16"]:
        if label in parts:
            return normalize_precision_label(label)
    name = path.name
    match = re.search(r"(Q[4568]_[A-Z0-9_]+|Q[4568]|F16|F32|BF16|INT8|FP16|FP32)", name, re.IGNORECASE)
    if match:
        return normalize_precision_label(match.group(1))
    return normalize_precision_label("unknown")


def detect_safetensors_folder_precision(folder: Path) -> tuple[str, str]:
    config = folder / "config.json"
    if config.exists():
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
            quant = data.get("quantization_config", {})
            if isinstance(quant, dict):
                for key in ["quant_method", "load_in_8bit", "load_in_4bit"]:
                    if key in quant:
                        value = quant[key]
                        if value is True and key == "load_in_8bit":
                            return normalize_precision_label("int8")
                        if value is True and key == "load_in_4bit":
                            return normalize_precision_label("int4")
                        if isinstance(value, str):
                            return normalize_precision_label(value)
            dtype = data.get("torch_dtype")
            if isinstance(dtype, str):
                return normalize_precision_label(dtype.replace("float", "fp"))
        except Exception:
            pass
    return detect_from_path(folder)
