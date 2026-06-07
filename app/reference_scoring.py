from __future__ import annotations

from .scoring import align_words, edit_distance, normalize_words, strict_words


def _chunk_metrics(reference: str, hypothesis: str, *, include_alignment: bool, max_alignment_cells: int) -> dict:
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)
    strict_ref = strict_words(reference)
    strict_hyp = strict_words(hypothesis)
    normalized_edits = edit_distance(ref_words, hyp_words)
    strict_edits = edit_distance(strict_ref, strict_hyp)
    cer_edits = edit_distance(reference, hypothesis)
    normalized_cer_edits = edit_distance(" ".join(ref_words), " ".join(hyp_words))
    alignment: list[dict] = []
    substitutions = insertions = deletions = 0
    alignment_too_large = len(ref_words) * max(1, len(hyp_words)) > max_alignment_cells
    if include_alignment and not alignment_too_large:
        alignment = align_words(ref_words, hyp_words)
        substitutions = sum(1 for item in alignment if item["op"] == "replace")
        insertions = sum(1 for item in alignment if item["op"] == "insert")
        deletions = sum(1 for item in alignment if item["op"] == "delete")
    return {
        "reference_word_count": len(ref_words),
        "hypothesis_word_count": len(hyp_words),
        "normalized_edits": normalized_edits,
        "strict_reference_word_count": len(strict_ref),
        "strict_edits": strict_edits,
        "reference_char_count": len(reference),
        "cer_edits": cer_edits,
        "normalized_reference_char_count": len(" ".join(ref_words)),
        "normalized_cer_edits": normalized_cer_edits,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "alignment": alignment,
        "alignment_too_large": alignment_too_large,
    }


def score_results_against_reference(
    results: dict,
    reference: dict,
    *,
    include_alignment: bool = True,
    max_alignment_cells: int = 12_000_000,
) -> dict:
    reference_by_chunk = {segment["chunk_id"]: segment.get("text", "") for segment in reference.get("segments", [])}
    scores: dict[str, dict] = {}
    for run in results.get("runs", []):
        candidate_id = run.get("model", {}).get("candidate_id", "")
        totals = {
            "reference_word_count": 0,
            "normalized_edits": 0,
            "strict_reference_word_count": 0,
            "strict_edits": 0,
            "reference_char_count": 0,
            "cer_edits": 0,
            "normalized_reference_char_count": 0,
            "normalized_cer_edits": 0,
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
        }
        alignments: list[dict] = []
        chunk_scores = []
        for chunk in run.get("transcript_chunks", []):
            chunk_id = chunk.get("chunk_id")
            metrics = _chunk_metrics(
                reference_by_chunk.get(chunk_id, ""),
                chunk.get("text", ""),
                include_alignment=include_alignment,
                max_alignment_cells=max_alignment_cells,
            )
            for key in totals:
                totals[key] += int(metrics[key])
            if metrics["alignment"]:
                alignments.extend(metrics["alignment"])
            chunk_scores.append(
                {
                    "chunk_id": chunk_id,
                    "normalized_wer": metrics["normalized_edits"] / max(1, metrics["reference_word_count"]),
                    "strict_wer": metrics["strict_edits"] / max(1, metrics["strict_reference_word_count"]),
                    "cer": metrics["cer_edits"] / max(1, metrics["reference_char_count"]),
                    "alignment_too_large": metrics["alignment_too_large"],
                }
            )
        scores[candidate_id] = {
            "normalized_wer": totals["normalized_edits"] / max(1, totals["reference_word_count"]),
            "strict_wer": totals["strict_edits"] / max(1, totals["strict_reference_word_count"]),
            "cer": totals["cer_edits"] / max(1, totals["reference_char_count"]),
            "normalized_cer": totals["normalized_cer_edits"] / max(1, totals["normalized_reference_char_count"]),
            "substitutions": totals["substitutions"],
            "insertions": totals["insertions"],
            "deletions": totals["deletions"],
            "alignment": alignments,
            "chunk_scores": chunk_scores,
            "alignment_mode": "omitted_for_large_chunks" if any(item["alignment_too_large"] for item in chunk_scores) else "included",
        }
    return scores
