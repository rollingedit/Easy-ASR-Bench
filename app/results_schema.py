from __future__ import annotations


REQUIRED_RESULT_KEYS = {
    "schema",
    "app_version",
    "source",
    "environment",
    "dependency_versions",
    "chunk_plan",
    "selected_models",
    "runs",
    "unsupported_models",
    "pairwise_differences",
    "errors",
}


def validate_results_schema(results: dict) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_RESULT_KEYS - set(results))
    if missing:
        errors.append("results missing keys: " + ", ".join(missing))
    if results.get("schema") != "easy_asr_bench.results.v1":
        errors.append("results schema must be easy_asr_bench.results.v1")
    if not isinstance(results.get("runs", []), list):
        errors.append("runs must be a list")
    for index, run in enumerate(results.get("runs", [])):
        if not isinstance(run, dict):
            errors.append(f"run {index} must be an object")
            continue
        for key in ["model", "transcript_chunks", "metrics", "errors"]:
            if key not in run:
                errors.append(f"run {index} missing {key}")
        metrics = run.get("metrics", {})
        if isinstance(metrics, dict):
            for key in ["provider", "peak_process_memory_mb", "peak_vram_mb"]:
                if key not in metrics:
                    errors.append(f"run {index} metrics missing {key}")
    return errors
