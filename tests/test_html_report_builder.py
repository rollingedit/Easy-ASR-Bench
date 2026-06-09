from app.html_report_builder import build_html_report


def minimal_results(chunk_count: int = 1) -> dict:
    chunks = [
        {
            "chunk_id": f"{index + 1:04d}",
            "start_seconds": float(index),
            "end_seconds": float(index + 1),
            "start_timestamp": f"00:00:{index:02d}.000",
            "end_timestamp": f"00:00:{index + 1:02d}.000",
        }
        for index in range(chunk_count)
    ]
    return {
        "source": {"name": "sample.wav", "sha256": "abc", "duration_seconds": chunk_count},
        "chunk_plan": {"chunks": chunks},
        "runs": [
            {
                "model": {"candidate_id": "m1", "display_name": "Model One", "precision": "fp32", "backend": "test"},
                "metrics": {"audio_seconds_per_wall_second": 1.0, "peak_process_memory_mb": 100, "peak_vram_mb": None, "vram_measurement_source": "unavailable"},
                "transcript_chunks": [{"chunk_id": chunk["chunk_id"], "text": "hello"} for chunk in chunks],
                "errors": [],
            }
        ],
        "pairwise_differences": {},
    }


def test_html_report_contains_reference_validation_guards():
    html = build_html_report(minimal_results())

    assert "easy_asr_bench.llm_reference.v1" in html
    assert "sourceMismatch" in html
    assert "duplicates" in html
    assert "timestampErrors" in html
    assert "LLM-corrected reference scores, not human ground truth" in html


def test_embedded_results_json_is_not_html_entity_escaped():
    html = build_html_report(minimal_results())

    assert "&quot;" not in html
    assert '<script type="application/json" id="results-json">{"source"' in html


def test_embedded_results_json_escapes_script_breakout():
    data = minimal_results()
    data["source"]["name"] = "</script><script>alert(1)</script>"

    html = build_html_report(data)

    embedded = html.split('<script type="application/json" id="results-json">', 1)[1].split("</script>", 1)[0]
    assert "</script>" not in embedded.lower()
    assert "\\u003c/script\\u003e" in embedded


def test_html_report_contains_chunk_pagination_for_long_reports():
    html = build_html_report(minimal_results(chunk_count=60))

    assert "const pageSize = 25" in html
    assert "chunkPage" in html
    assert "Chunk page ${chunkPage+1} / ${totalPages}" in html


def test_html_report_guards_browser_scoring_for_huge_references():
    html = build_html_report(minimal_results())

    assert "const maxBrowserScoreCells" in html
    assert "too large for browser WER/CER scoring" in html


def test_html_report_can_display_precomputed_reference_scores():
    data = minimal_results()
    data["reference_scores"] = {"m1": {"normalized_wer": 0.0, "strict_wer": 0.0, "cer": 0.0, "substitutions": 0, "insertions": 0, "deletions": 0}}
    data["reference"] = {
        "schema": "easy_asr_bench.llm_reference.v1",
        "source_sha256": "abc",
        "reference_type": "llm_corrected_reference",
        "segments": [{"chunk_id": "0001", "start_seconds": 0.0, "end_seconds": 1.0, "text": "hello world", "uncertain": ["AI-assisted correction"]}],
        "global_notes": ["fixture reference"],
    }
    data["score_note"] = "LLM-corrected reference scores are not human ground truth."

    html = build_html_report(data)

    assert "precomputedReferenceScores" in html
    assert "Loaded LLM-Corrected Reference" in html
    assert "hello world" in html
    assert "Loaded precomputed LLM-corrected reference scores" in html
    assert "renderScoreboard(latestScores)" in html


def test_html_report_separates_balanced_and_runtime_only_rankings():
    data = minimal_results()
    data["runtime_rankings"] = {
        "note": "Runtime rankings do not measure transcript quality.",
        "rows": [
            {
                "runtime_rank": 1,
                "display_name": "Model One",
                "speed_percentile": 1.0,
                "memory_percentile_inverse": 1.0,
                "speed_audio_seconds_per_wall_second": 1.0,
                "peak_process_memory_mb": 100,
                "peak_vram_mb": None,
            }
        ],
    }

    html = build_html_report(data)

    assert "Speed and Memory Only" in html
    assert "Accuracy is not scored yet because no corrected reference is loaded" in html
    assert "does not measure transcript quality" in html


def test_html_report_accepts_scored_report_payload_directly():
    data = minimal_results()
    scored = {
        "schema": "easy_asr_bench.scored_report.v1",
        "status": "scored",
        "score_type": "llm_corrected_reference",
        "score_note": "Scores are AI-assisted.",
        "results": data,
        "reference": {
            "schema": "easy_asr_bench.llm_reference.v1",
            "source_sha256": "abc",
            "reference_type": "llm_corrected_reference",
            "segments": [{"chunk_id": "0001", "start_seconds": 0.0, "end_seconds": 1.0, "text": "real reference", "uncertain": []}],
        },
        "scores": {"m1": {"normalized_wer": 0.0, "strict_wer": 0.0, "cer": 0.0, "substitutions": 0, "insertions": 0, "deletions": 0, "balanced_rank": 1}},
    }

    html = build_html_report(scored)

    assert '"reference_scores"' in html
    assert "real reference" in html
    assert "Accuracy Rank" in html
    assert "Loaded precomputed LLM-corrected reference scores" in html
    assert "System RAM Peak" in html
    assert "VRAM / GPU Memory Peak" in html
    assert "do not add it to RAM as a total" in html
