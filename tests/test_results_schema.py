from app.results_schema import validate_results_schema


def minimal_results():
    return {
        "schema": "easy_asr_bench.results.v1",
        "app_version": "0.0.0",
        "created_local": "2026-06-10 00:00:00",
        "source": {"path": "Input/audio.wav", "name": "audio.wav", "sha256": "abc", "duration_seconds": 1.0},
        "environment": {},
        "dependency_versions": {},
        "adapter_versions": {},
        "chunk_plan": {
            "sample_rate": 16000,
            "source_audio_seconds": 1.0,
            "chunks": [{"chunk_id": "0001", "index": 0, "start_seconds": 0.0, "end_seconds": 1.0}],
        },
        "selected_models": [],
        "runs": [
            {
                "model": {"candidate_id": "model", "display_name": "Model", "adapter_name": "fixture"},
                "transcript_chunks": [{"chunk_id": "0001", "start_seconds": 0.0, "end_seconds": 1.0, "text": "hello"}],
                "metrics": {"provider": "fixture", "peak_process_memory_mb": 1, "peak_vram_mb": None, "vram_measurement_source": "unavailable"},
                "errors": [],
            }
        ],
        "unsupported_models": [],
        "pairwise_differences": {},
        "runtime_rankings": {"schema": "easy_asr_bench.runtime_rankings.v1", "rows": []},
        "errors": [],
    }


def test_results_schema_accepts_canonical_result_shape():
    assert validate_results_schema(minimal_results()) == []


def test_results_schema_rejects_missing_runtime_metrics():
    results = minimal_results()
    del results["runs"][0]["metrics"]["peak_vram_mb"]

    assert any("peak_vram_mb" in error for error in validate_results_schema(results))


def test_results_schema_rejects_missing_vram_source():
    results = minimal_results()
    del results["runs"][0]["metrics"]["vram_measurement_source"]

    assert any("vram_measurement_source" in error for error in validate_results_schema(results))


def test_results_schema_rejects_missing_source_identity():
    results = minimal_results()
    del results["source"]["sha256"]

    assert any("source missing sha256" in error for error in validate_results_schema(results))


def test_results_schema_rejects_invalid_chunk_plan_timing():
    results = minimal_results()
    results["chunk_plan"]["chunks"][0]["end_seconds"] = -1

    assert any("end_seconds must be >= start_seconds" in error for error in validate_results_schema(results))


def test_results_schema_rejects_transcript_chunk_not_in_plan():
    results = minimal_results()
    results["runs"][0]["transcript_chunks"][0]["chunk_id"] = "9999"

    assert any("not in chunk_plan" in error for error in validate_results_schema(results))


def test_results_schema_rejects_malformed_structured_run_error():
    results = minimal_results()
    results["runs"][0]["errors"].append({"status": "chunk_failed", "stage": "chunk_inference", "message": "boom"})

    assert any("error_type" in error for error in validate_results_schema(results))
