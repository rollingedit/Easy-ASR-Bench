from __future__ import annotations

from .scoring import align_words, balanced_score, edit_distance, normalize_words, strict_words


def _higher_is_better_percentiles(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    if len(values) == 1:
        return {next(iter(values)): 1.0}
    ordered = sorted(values.items(), key=lambda item: item[1])
    denominator = max(1, len(ordered) - 1)
    return {candidate_id: rank / denominator for rank, (candidate_id, _value) in enumerate(ordered)}


def _lower_is_better_percentiles(values: dict[str, float]) -> dict[str, float]:
    higher = _higher_is_better_percentiles({candidate_id: -value for candidate_id, value in values.items()})
    return higher


def runtime_ranking_inputs(results: dict) -> tuple[dict[str, float], dict[str, float]]:
    speed: dict[str, float] = {}
    memory: dict[str, float] = {}
    for run in results.get("runs", []):
        candidate_id = run.get("model", {}).get("candidate_id", "")
        if not candidate_id:
            continue
        metrics = run.get("metrics", {})
        try:
            speed_value = float(metrics.get("audio_seconds_per_wall_second", 0) or 0)
        except (TypeError, ValueError):
            speed_value = 0.0
        if speed_value > 0:
            speed[candidate_id] = speed_value
        memory_values = []
        for key in ("peak_process_memory_mb", "peak_vram_mb"):
            try:
                value = metrics.get(key)
                if value is not None:
                    memory_values.append(float(value))
            except (TypeError, ValueError):
                continue
        if memory_values:
            memory[candidate_id] = sum(memory_values)
    return speed, memory


def runtime_rankings(results: dict) -> dict:
    speed, memory = runtime_ranking_inputs(results)
    speed_percentiles = _higher_is_better_percentiles(speed)
    memory_percentiles = _lower_is_better_percentiles(memory)
    rows = []
    for run in results.get("runs", []):
        candidate_id = run.get("model", {}).get("candidate_id", "")
        metrics = run.get("metrics", {})
        rows.append(
            {
                "candidate_id": candidate_id,
                "display_name": run.get("model", {}).get("display_name", candidate_id),
                "speed_audio_seconds_per_wall_second": metrics.get("audio_seconds_per_wall_second"),
                "peak_process_memory_mb": metrics.get("peak_process_memory_mb"),
                "peak_vram_mb": metrics.get("peak_vram_mb"),
                "speed_percentile": speed_percentiles.get(candidate_id),
                "memory_percentile_inverse": memory_percentiles.get(candidate_id),
                "rank_basis": "runtime_only_no_quality_reference",
            }
        )
    rows.sort(
        key=lambda row: (
            row["speed_percentile"] if row["speed_percentile"] is not None else -1,
            row["memory_percentile_inverse"] if row["memory_percentile_inverse"] is not None else -1,
        ),
        reverse=True,
    )
    for index, row in enumerate(rows, 1):
        row["runtime_rank"] = index
    return {
        "schema": "easy_asr_bench.runtime_rankings.v1",
        "note": "Runtime rankings do not measure transcript quality. Use LLM-corrected or human reference scores for accuracy-aware ranking.",
        "rows": rows,
    }


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
    speed, memory = runtime_ranking_inputs(results)
    speed_percentiles = _higher_is_better_percentiles(speed)
    memory_percentiles = _lower_is_better_percentiles(memory)
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
        normalized_wer = totals["normalized_edits"] / max(1, totals["reference_word_count"])
        quality = max(0.0, 1.0 - normalized_wer)
        overall_score = balanced_score(
            quality,
            speed_percentiles.get(candidate_id, 0.0),
            memory_percentiles.get(candidate_id, 0.0),
        )
        scores[candidate_id] = {
            "normalized_wer": normalized_wer,
            "strict_wer": totals["strict_edits"] / max(1, totals["strict_reference_word_count"]),
            "cer": totals["cer_edits"] / max(1, totals["reference_char_count"]),
            "normalized_cer": totals["normalized_cer_edits"] / max(1, totals["normalized_reference_char_count"]),
            "quality_component": quality,
            "speed_percentile": speed_percentiles.get(candidate_id),
            "memory_percentile_inverse": memory_percentiles.get(candidate_id),
            "balanced_score": overall_score,
            "balanced_score_note": "70% LLM-corrected reference quality, 20% speed percentile, 10% inverse RAM/VRAM percentile.",
            "substitutions": totals["substitutions"],
            "insertions": totals["insertions"],
            "deletions": totals["deletions"],
            "alignment": alignments,
            "chunk_scores": chunk_scores,
            "alignment_mode": "omitted_for_large_chunks" if any(item["alignment_too_large"] for item in chunk_scores) else "included",
        }
    for rank, (_candidate_id, score) in enumerate(sorted(scores.items(), key=lambda item: item[1]["balanced_score"], reverse=True), 1):
        score["balanced_rank"] = rank
    return scores
