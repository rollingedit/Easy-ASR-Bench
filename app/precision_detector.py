from __future__ import annotations

import json
import re
from pathlib import Path


def normalize_precision_label(raw: str | None) -> tuple[str, str]:
    label = (raw or "unknown").strip()
    lower = label.lower()
    if lower in {"int2", "q2", "q2_k"}:
        return label, "2-bit / Q2"
    if lower in {"int3", "q3", "q3_k_l", "q3_k_m", "q3_k_s"}:
        return label, "3-bit / Q3"
    if lower in {"int4", "q4", "q4_k_m", "q4_k_s", "q4_k_l", "q4_0", "q4_1", "iq4_nl", "iq4_xs", "fp4", "float4", "nf4", "nvfp4", "nvp4"}:
        return label, "4-bit / Q4"
    if lower in {"int5", "q5", "q5_k_m", "q5_k_s", "q5_k_l", "q5_0", "q5_1"}:
        return label, "5-bit / Q5"
    if lower in {"int6", "q6", "q6_k"}:
        return label, "6-bit / Q6"
    if lower in {"int8", "q8", "q8_0", "i8", "fp8", "float8", "e4m3", "e5m2", "bf8", "bfloat8"}:
        return label, "8-bit / INT8 / Q8"
    if lower.startswith("iq1_"):
        return label, "1-bit / IQ1"
    if lower.startswith("iq2_"):
        return label, "2-bit / IQ2"
    if lower.startswith("iq3_"):
        return label, "3-bit / IQ3"
    if lower in {"fp16", "f16", "float16", "fp16w"}:
        return label, "16-bit / FP16-family"
    if lower in {"bf16", "bfloat16", "bfloat16_t"}:
        return label, "16-bit / BF16"
    if lower in {"fp32", "f32", "float32"}:
        return label, "32-bit / FP32"
    return label, "Unknown precision"


def detect_from_path(path: Path) -> tuple[str, str]:
    parts = [part.lower() for part in path.parts]
    for label in ["int4", "int5", "int6", "int8", "fp4", "nf4", "nvfp4", "nvp4", "fp8", "bf8", "fp16w", "fp32", "f32", "float32", "fp16", "f16", "float16", "bf16", "bfloat16"]:
        if label in parts:
            return normalize_precision_label(label)
    name = path.name
    match = re.search(
        r"(IQ[1-4]_[A-Z0-9_]+|Q[2-8]_[A-Z0-9_]+|Q[2-8]|F16|F32|BF16|BF8|BFLOAT16|BFLOAT8|INT[2-8]|NF4|NVFP4|NVP4|FP4|FP8|FP16|FP32|E4M3|E5M2)",
        name,
        re.IGNORECASE,
    )
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
                for key in ["quant_method", "load_in_8bit", "load_in_4bit", "bnb_4bit_quant_type", "quant_type"]:
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
                dtype_lower = dtype.strip().lower()
                if dtype_lower.startswith("bfloat"):
                    return normalize_precision_label(dtype_lower)
                return normalize_precision_label(dtype_lower.replace("float", "fp"))
        except Exception:
            pass
    return detect_from_path(folder)


def safetensor_index_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.glob("*.json")
        if ".safetensors.index" in path.name.lower()
    )


def indexed_safetensor_missing_files(folder: Path) -> list[str]:
    indexes = safetensor_index_files(folder)
    if not indexes:
        return []
    missing: set[str] = set()
    errors: list[str] = []
    for index in indexes:
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
        except Exception:
            errors.append(f"{index.name} (parseable JSON)")
            continue
        weight_map = data.get("weight_map", {})
        if not isinstance(weight_map, dict):
            errors.append(f"{index.name} weight_map")
            continue
        for name in weight_map.values():
            if isinstance(name, str) and not (folder / name).exists():
                missing.add(name)
    return sorted([*missing, *errors])
