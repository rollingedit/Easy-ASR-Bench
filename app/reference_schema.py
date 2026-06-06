from __future__ import annotations


def validate_llm_reference(reference: dict, results: dict) -> list[str]:
    errors: list[str] = []
    if reference.get("schema") != "easy_asr_bench.llm_reference.v1":
        errors.append("Reference schema must be easy_asr_bench.llm_reference.v1.")
    if reference.get("reference_type") != "llm_corrected_reference":
        errors.append("reference_type must be llm_corrected_reference.")
    expected = {chunk["chunk_id"] for chunk in results.get("chunk_plan", {}).get("chunks", [])}
    seen = {segment.get("chunk_id") for segment in reference.get("segments", [])}
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing:
        errors.append("Reference is missing chunk IDs: " + ", ".join(missing))
    if extra:
        errors.append("Reference has unknown chunk IDs: " + ", ".join(extra))
    for segment in reference.get("segments", []):
        if not isinstance(segment.get("text"), str):
            errors.append(f"Chunk {segment.get('chunk_id')} text must be a string.")
    return errors
