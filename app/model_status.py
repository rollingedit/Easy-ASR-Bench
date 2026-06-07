from __future__ import annotations

from .adapters.base import ModelCandidate


STATUS_LABELS = {
    "runnable_asr": "Runnable ASR",
    "needs_dependency_install": "Needs dependency install",
    "asr_probe_required": "ASR probe required",
    "reference_llm": "Reference/correction LLM",
    "recognized_incomplete": "Recognized incomplete",
    "recognized_ambiguous_not_runnable": "Recognized ambiguous",
    "unsafe_blocked": "Unsafe blocked",
    "unsupported_llm_format": "Unsupported LLM format",
    "unknown_inspection_only": "Unknown inspection-only",
    "unsupported_format": "Unsupported format",
}


def model_status(candidate: ModelCandidate) -> str:
    if candidate.runnable and candidate.category == "asr":
        return "runnable_asr"
    if candidate.runnable_after_dependency_install:
        return "needs_dependency_install"
    category = candidate.metadata.get("model_status") or candidate.category
    if category in STATUS_LABELS:
        return str(category)
    if category == "unsupported_llm":
        return "unsupported_llm_format"
    if category == "recognized_unsupported_asr":
        if candidate.missing_files:
            return "recognized_incomplete"
        return "unsupported_format"
    if "pickle" in " ".join(candidate.warnings).lower() or candidate.path.suffix.lower() == ".pt":
        return "unsafe_blocked"
    if candidate.missing_files:
        return "recognized_incomplete"
    return "unknown_inspection_only"


def model_status_label(candidate: ModelCandidate) -> str:
    return STATUS_LABELS[model_status(candidate)]


def candidate_reason(candidate: ModelCandidate) -> str:
    parts = []
    status = model_status_label(candidate)
    parts.append(status)
    parts.extend(candidate.warnings)
    if candidate.missing_files:
        parts.append(f"Missing: {', '.join(candidate.missing_files)}")
    if candidate.help_text:
        parts.append(candidate.help_text)
    return "; ".join(parts)
