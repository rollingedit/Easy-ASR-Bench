from app.reference_import import extract_reference_json, import_llm_reference
from app.reference_scoring import score_results_against_reference


def make_results(chunk_count: int = 2, words_per_chunk: int = 3) -> dict:
    chunks = [
        {"chunk_id": f"{index:04d}", "start_seconds": float(index), "end_seconds": float(index + 1)}
        for index in range(chunk_count)
    ]
    text = " ".join(f"word{word}" for word in range(words_per_chunk))
    return {
        "source": {"sha256": "abc"},
        "chunk_plan": {"chunks": chunks},
        "runs": [
            {
                "model": {"candidate_id": "model_a", "display_name": "Model A"},
                "transcript_chunks": [{"chunk_id": chunk["chunk_id"], "text": text} for chunk in chunks],
            }
        ],
    }


def make_reference(results: dict) -> dict:
    return {
        "schema": "easy_asr_bench.llm_reference.v1",
        "source_sha256": "abc",
        "reference_type": "llm_corrected_reference",
        "segments": [
            {
                "chunk_id": chunk["chunk_id"],
                "start_seconds": chunk["start_seconds"],
                "end_seconds": chunk["end_seconds"],
                "text": "word0 word1 word2",
                "uncertain": [],
            }
            for chunk in results["chunk_plan"]["chunks"]
        ],
        "global_notes": [],
    }


def test_extract_reference_json_accepts_fenced_or_full_llm_response():
    parsed = extract_reference_json('Here is the JSON:\n```json\n{"schema":"x"}\n```')

    assert parsed == {"schema": "x"}


def test_import_llm_reference_validates_and_scores():
    results = make_results()
    reference = make_reference(results)

    scored = import_llm_reference(results, reference)

    assert scored["status"] == "scored"
    assert scored["score_type"] == "llm_corrected_reference"
    assert scored["scores"]["model_a"]["normalized_wer"] == 0


def test_import_llm_reference_rejects_invalid_source_hash():
    results = make_results()
    reference = make_reference(results)
    reference["source_sha256"] = "wrong"

    scored = import_llm_reference(results, reference)

    assert scored["status"] == "invalid"
    assert any("source_sha256" in error for error in scored["errors"])


def test_large_reference_scoring_runs_per_chunk_without_global_browser_dp():
    results = make_results(chunk_count=500, words_per_chunk=100)
    reference = make_reference(results)
    for segment in reference["segments"]:
        segment["text"] = " ".join(f"word{word}" for word in range(100))

    scores = score_results_against_reference(results, reference, include_alignment=False)

    assert scores["model_a"]["normalized_wer"] == 0
    assert len(scores["model_a"]["chunk_scores"]) == 500
