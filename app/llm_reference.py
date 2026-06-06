from __future__ import annotations

from pathlib import Path

from .adapters.base import ModelCandidate
from .adapters.gguf_llm_reference import GGUFLLMReferenceAdapter
from .config import save_config


EXTERNAL_LLM_GUIDE = """\
Manual LLM-corrected reference workflow

Use this when you want ChatGPT, Claude, or another external LLM to judge/correct
the transcript instead of a local GGUF model.

1. Run the benchmark normally.
2. Open the report folder in Output.
3. Open results.txt or results_llm_prompt_part_001.txt.
4. Paste the LLM-corrected reference instructions into your LLM.
5. Copy the returned JSON with schema easy_asr_bench.llm_reference.v1.
6. Open compare.html.
7. Paste the JSON into the reference box and score the models.

The LLM reference is AI-assisted review, not human ground truth. Use a strong
model and inspect uncertain sections before making final model decisions.
"""


def llm_reference_config(config: dict) -> dict:
    section = config.setdefault("llm_reference", {})
    section.setdefault("custom_model_paths", [])
    section.setdefault("manual_external_llm_guide", True)
    section.setdefault("auto_scan_saved_paths", True)
    return section


def scan_custom_reference_llms(config: dict) -> list[ModelCandidate]:
    section = llm_reference_config(config)
    if not section.get("auto_scan_saved_paths", True):
        return []
    adapter = GGUFLLMReferenceAdapter()
    candidates: list[ModelCandidate] = []
    for raw_path in section.get("custom_model_paths", []):
        if isinstance(raw_path, str) and raw_path.strip():
            candidates.extend(adapter.discover_path(Path(raw_path.strip())))
    return candidates


def merge_reference_llms(*groups: list[ModelCandidate]) -> list[ModelCandidate]:
    merged: dict[str, ModelCandidate] = {}
    for group in groups:
        for candidate in group:
            if candidate.category != "reference_llm":
                continue
            key = str(candidate.path.resolve()).lower()
            merged[key] = candidate
    return list(merged.values())


def save_custom_reference_path(config_path: Path, config: dict, raw_path: str) -> list[ModelCandidate]:
    path = Path(raw_path.strip().strip('"')).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    adapter = GGUFLLMReferenceAdapter()
    candidates = adapter.discover_path(path)
    if not candidates:
        raise ValueError("No .gguf reference LLM was found at that file or inside that folder.")
    section = llm_reference_config(config)
    resolved = str(path.resolve())
    existing = {str(Path(item).expanduser()).lower() for item in section["custom_model_paths"] if isinstance(item, str)}
    if resolved.lower() not in existing:
        section["custom_model_paths"].append(resolved)
        save_config(config_path, config)
    return candidates


def print_external_llm_guide() -> None:
    print()
    print(EXTERNAL_LLM_GUIDE)
