from app.reference_scoring import runtime_rankings
from app.scoring import pairwise_metrics, score_against_reference


def test_pairwise_metrics_exact_match():
    metrics = pairwise_metrics("hello world", "hello world")
    assert metrics["normalized_wer_like_difference"] == 0


def test_reference_scoring_detects_difference():
    metrics = score_against_reference("hello world", "hello there")
    assert metrics["normalized_wer"] > 0


def test_runtime_rankings_are_labeled_runtime_only():
    rankings = runtime_rankings(
        {
            "runs": [
                {
                    "model": {"candidate_id": "fast", "display_name": "Fast"},
                    "metrics": {"audio_seconds_per_wall_second": 10, "peak_process_memory_mb": 200, "peak_vram_mb": 100},
                },
                {
                    "model": {"candidate_id": "slow", "display_name": "Slow"},
                    "metrics": {"audio_seconds_per_wall_second": 1, "peak_process_memory_mb": 100, "peak_vram_mb": None},
                },
            ]
        }
    )

    assert rankings["schema"] == "easy_asr_bench.runtime_rankings.v1"
    assert "do not measure transcript quality" in rankings["note"]
    assert {row["rank_basis"] for row in rankings["rows"]} == {"runtime_only_no_quality_reference"}
