from app.results_schema import validate_results_schema


def minimal_results():
    return {
        "schema": "easy_asr_bench.results.v1",
        "app_version": "0.0.0",
        "source": {},
        "environment": {},
        "dependency_versions": {},
        "chunk_plan": {},
        "selected_models": [],
        "runs": [
            {
                "model": {},
                "transcript_chunks": [],
                "metrics": {"provider": "fixture", "peak_process_memory_mb": 1, "peak_vram_mb": None},
                "errors": [],
            }
        ],
        "unsupported_models": [],
        "pairwise_differences": {},
        "errors": [],
    }


def test_results_schema_accepts_canonical_result_shape():
    assert validate_results_schema(minimal_results()) == []


def test_results_schema_rejects_missing_runtime_metrics():
    results = minimal_results()
    del results["runs"][0]["metrics"]["peak_vram_mb"]

    assert any("peak_vram_mb" in error for error in validate_results_schema(results))
