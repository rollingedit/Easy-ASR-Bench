from app.html_report_builder import build_html_report


def large_results(chunk_count: int = 500) -> dict:
    chunks = [
        {
            "chunk_id": f"{index + 1:04d}",
            "start_seconds": float(index * 30),
            "end_seconds": float((index + 1) * 30),
            "start_timestamp": "00:00:00.000",
            "end_timestamp": "00:00:30.000",
        }
        for index in range(chunk_count)
    ]
    runs = [
        {
            "model": {"candidate_id": "model_a", "display_name": "Model A", "backend": "fixture", "precision": "fp32"},
            "metrics": {"peak_process_memory_mb": 100, "peak_vram_mb": None},
            "transcript_chunks": [{"chunk_id": chunk["chunk_id"], "text": "hello world " * 50} for chunk in chunks],
            "errors": [],
        },
        {
            "model": {"candidate_id": "model_b", "display_name": "Model B", "backend": "fixture", "precision": "int8"},
            "metrics": {"peak_process_memory_mb": 80, "peak_vram_mb": None},
            "transcript_chunks": [{"chunk_id": chunk["chunk_id"], "text": "hello word " * 50} for chunk in chunks],
            "errors": ["fixture warning"],
        },
    ]
    return {
        "source": {"name": "large.wav", "duration_seconds": chunk_count * 30, "sha256": "fixture"},
        "chunk_plan": {"chunks": chunks},
        "runs": runs,
        "unsupported_models": [{"display_name": "Skipped model", "help_text": "fixture skipped"}],
        "pairwise_differences": {"model_a vs model_b": {"normalized_wer_like_difference": 0.1, "cer_difference": 0.05}},
    }


def test_compare_html_large_500_chunk_fixture_is_self_contained_and_paginated():
    html = build_html_report(large_results())

    assert "<!doctype html>" in html
    assert "https://" not in html
    assert "http://" not in html
    assert "const pageSize = 25" in html
    assert "const transcriptPageSize = 5000" in html
    assert "function renderChunks" in html
    assert "chunks.slice(chunkPage * pageSize" in html
    assert "function renderTranscriptTextPage" in html
    assert "fixture warning" in html
    assert "Skipped model" in html
