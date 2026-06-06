from app.scoring import pairwise_metrics, score_against_reference


def test_pairwise_metrics_exact_match():
    metrics = pairwise_metrics("hello world", "hello world")
    assert metrics["normalized_wer_like_difference"] == 0


def test_reference_scoring_detects_difference():
    metrics = score_against_reference("hello world", "hello there")
    assert metrics["normalized_wer"] > 0
