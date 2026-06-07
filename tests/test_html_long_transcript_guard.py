from app.html_report_builder import build_html_report


def long_results():
    chunks = [
        {
            "chunk_id": f"{index:04d}",
            "start_seconds": float(index),
            "end_seconds": float(index + 1),
            "start_timestamp": "00:00:00.000",
            "end_timestamp": "00:00:01.000",
        }
        for index in range(60)
    ]
    runs = []
    for model_index in range(3):
        runs.append(
            {
                "model": {
                    "candidate_id": f"model_{model_index}",
                    "display_name": f"Model {model_index}",
                    "backend": "fixture",
                    "precision": "fp32",
                },
                "transcript_chunks": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "start_seconds": chunk["start_seconds"],
                        "end_seconds": chunk["end_seconds"],
                        "text": "word " * 2000,
                    }
                    for chunk in chunks
                ],
                "metrics": {},
                "errors": [],
            }
        )
    return {
        "source": {"name": "long.wav", "duration_seconds": 21600, "sha256": "abc"},
        "chunk_plan": {"chunks": chunks},
        "runs": runs,
        "pairwise_differences": {},
    }


def test_html_uses_transcript_and_alignment_paging_helpers():
    html = build_html_report(long_results())

    assert "const transcriptPageSize = 5000" in html
    assert "const alignmentPageSize = 500" in html
    assert "function renderTranscriptTextPage" in html
    assert "function renderAlignmentPage" in html
    assert "renderAlignment(s.alignment)" not in html
    assert "slice(page * alignmentPageSize" in html
    assert "Transcript page" in html
    assert "Alignment page" in html
