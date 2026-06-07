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
                "metrics": {"audio_seconds_per_wall_second": 1.0, "peak_process_memory_mb": 100, "peak_vram_mb": None},
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


def test_html_report_contains_chunk_pagination_for_long_reports():
    html = build_html_report(minimal_results(chunk_count=60))

    assert "const pageSize = 25" in html
    assert "chunkPage" in html
    assert "Chunk page ${chunkPage+1} / ${totalPages}" in html
