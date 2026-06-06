from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


DEPENDENCY_GROUPS = {
    "core": ["numpy", "soundfile", "librosa", "imageio_ffmpeg", "psutil", "jiwer"],
    "onnx": ["onnxruntime", "tokenizers", "jinja2"],
    "transformers_cpu": ["torch", "transformers", "safetensors"],
    "llama_cpp": ["llama_cpp"],
}


REQUIREMENT_FILES = {
    "core": "requirements/core.txt",
    "onnx": "requirements/onnx.txt",
    "transformers_cpu": "requirements/transformers_cpu.txt",
    "llama_cpp": "requirements/llama_cpp.txt",
}


def missing_modules(group: str) -> list[str]:
    return [module for module in DEPENDENCY_GROUPS.get(group, []) if importlib.util.find_spec(module) is None]


def group_available(group: str) -> bool:
    return not missing_modules(group)


def install_group(group: str, project_root: Path) -> None:
    req = project_root / REQUIREMENT_FILES[group]
    if not req.exists():
        raise FileNotFoundError(f"Missing dependency requirement file: {req}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])


def dependency_status() -> dict[str, dict]:
    return {group: {"available": group_available(group), "missing": missing_modules(group)} for group in DEPENDENCY_GROUPS}
