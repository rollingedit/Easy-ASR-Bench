from __future__ import annotations


def validate_llm_reference(reference: dict, results: dict) -> list[str]:
    errors: list[str] = []
    if reference.get("schema") != "easy_asr_bench.llm_reference.v1":
        errors.append("Reference schema must be easy_asr_bench.llm_reference.v1.")
    if reference.get("reference_type") != "llm_corrected_reference":
        errors.append("reference_type must be llm_corrected_reference.")
    expected_source = results.get("source", {}).get("sha256")
    reference_source = reference.get("source_sha256")
    if expected_source and reference_source and reference_source != expected_source:
        errors.append("Reference source_sha256 does not match results source sha256.")
    expected = {chunk["chunk_id"] for chunk in results.get("chunk_plan", {}).get("chunks", [])}
    segments = reference.get("segments", [])
    seen_list = [segment.get("chunk_id") for segment in segments]
    seen = set(seen_list)
    duplicates = sorted({chunk_id for chunk_id in seen_list if seen_list.count(chunk_id) > 1})
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing:
        errors.append("Reference is missing chunk IDs: " + ", ".join(missing))
    if extra:
        errors.append("Reference has unknown chunk IDs: " + ", ".join(extra))
    if duplicates:
        errors.append("Reference has duplicate chunk IDs: " + ", ".join(duplicates))
    by_id = {chunk["chunk_id"]: chunk for chunk in results.get("chunk_plan", {}).get("chunks", [])}
    for segment in segments:
        if not isinstance(segment.get("text"), str):
            errors.append(f"Chunk {segment.get('chunk_id')} text must be a string.")
        if not isinstance(segment.get("uncertain", []), list):
            errors.append(f"Chunk {segment.get('chunk_id')} uncertain must be a list.")
        expected_chunk = by_id.get(segment.get("chunk_id"))
        if expected_chunk:
            for key in ["start_seconds", "end_seconds"]:
                if abs(float(segment.get(key, -999999)) - float(expected_chunk[key])) > 0.01:
                    errors.append(f"Chunk {segment.get('chunk_id')} {key} does not match results.")
    if not isinstance(reference.get("global_notes", []), list):
        errors.append("global_notes must be a list.")
    return errors
