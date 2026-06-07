from __future__ import annotations

import json
import re

from .reference_schema import validate_llm_reference
from .reference_scoring import score_results_against_reference


def extract_reference_json(text: str) -> dict:
    raw = str(text or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    if not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    return json.loads(raw)


def import_llm_reference(results: dict, reference_text_or_json: str | dict) -> dict:
    reference = reference_text_or_json if isinstance(reference_text_or_json, dict) else extract_reference_json(reference_text_or_json)
    errors = validate_llm_reference(reference, results)
    if errors:
        return {"status": "invalid", "errors": errors, "results": results, "reference": reference}
    scores = score_results_against_reference(results, reference)
    return {
        "schema": "easy_asr_bench.scored_report.v1",
        "status": "scored",
        "score_type": "llm_corrected_reference",
        "score_note": "LLM-corrected reference scores are not human ground truth.",
        "results": results,
        "reference": reference,
        "scores": scores,
    }
