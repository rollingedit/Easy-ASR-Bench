from __future__ import annotations


REQUIRED_RESULT_KEYS = {
    "schema",
    "app_version",
    "created_local",
    "source",
    "environment",
    "dependency_versions",
    "adapter_versions",
    "chunk_plan",
    "selected_models",
    "runs",
    "unsupported_models",
    "pairwise_differences",
    "runtime_rankings",
    "errors",
}


def _require_object(parent: dict, key: str, errors: list[str], path: str) -> dict:
    value = parent.get(key)
    if not isinstance(value, dict):
        errors.append(f"{path}.{key} must be an object")
        return {}
    return value


def _require_list(parent: dict, key: str, errors: list[str], path: str) -> list:
    value = parent.get(key)
    if not isinstance(value, list):
        errors.append(f"{path}.{key} must be a list")
        return []
    return value


def _require_keys(value: dict, keys: list[str], errors: list[str], path: str) -> None:
    for key in keys:
        if key not in value:
            errors.append(f"{path} missing {key}")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_chunk_plan_schema(chunk_plan: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(chunk_plan, dict):
        return ["chunk_plan must be an object"]
    _require_keys(chunk_plan, ["sample_rate", "source_audio_seconds", "chunks"], errors, "chunk_plan")
    if "sample_rate" in chunk_plan and not isinstance(chunk_plan["sample_rate"], int):
        errors.append("chunk_plan.sample_rate must be an integer")
    if "source_audio_seconds" in chunk_plan and not _is_number(chunk_plan["source_audio_seconds"]):
        errors.append("chunk_plan.source_audio_seconds must be a number")
    chunks = _require_list(chunk_plan, "chunks", errors, "chunk_plan")
    seen: set[str] = set()
    for index, chunk in enumerate(chunks):
        path = f"chunk_plan.chunks[{index}]"
        if not isinstance(chunk, dict):
            errors.append(f"{path} must be an object")
            continue
        _require_keys(chunk, ["chunk_id", "index", "start_seconds", "end_seconds"], errors, path)
        chunk_id = str(chunk.get("chunk_id", ""))
        if not chunk_id:
            errors.append(f"{path}.chunk_id must be non-empty")
        if chunk_id in seen:
            errors.append(f"{path}.chunk_id duplicates {chunk_id}")
        seen.add(chunk_id)
        for key in ["start_seconds", "end_seconds"]:
            if key in chunk and not _is_number(chunk[key]):
                errors.append(f"{path}.{key} must be a number")
        if _is_number(chunk.get("start_seconds")) and _is_number(chunk.get("end_seconds")) and chunk["end_seconds"] < chunk["start_seconds"]:
            errors.append(f"{path}.end_seconds must be >= start_seconds")
    return errors


def validate_run_error_schema(error, path: str = "error") -> list[str]:
    if isinstance(error, str):
        return []
    if not isinstance(error, dict):
        return [f"{path} must be an object or string"]
    errors: list[str] = []
    _require_keys(error, ["status", "stage", "message"], errors, path)
    if "status" in error and not isinstance(error["status"], str):
        errors.append(f"{path}.status must be a string")
    if "stage" in error and not isinstance(error["stage"], str):
        errors.append(f"{path}.stage must be a string")
    if "message" in error and not isinstance(error["message"], str):
        errors.append(f"{path}.message must be a string")
    if error.get("status") == "chunk_failed":
        _require_keys(error, ["chunk_id", "start_seconds", "end_seconds", "error_type"], errors, path)
    return errors


def validate_results_schema(results: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(results, dict):
        return ["results must be an object"]
    missing = sorted(REQUIRED_RESULT_KEYS - set(results))
    if missing:
        errors.append("results missing keys: " + ", ".join(missing))
    if results.get("schema") != "easy_asr_bench.results.v1":
        errors.append("results schema must be easy_asr_bench.results.v1")
    source = _require_object(results, "source", errors, "results")
    _require_keys(source, ["path", "name", "sha256", "duration_seconds"], errors, "source")
    if "duration_seconds" in source and not _is_number(source["duration_seconds"]):
        errors.append("source.duration_seconds must be a number")
    errors.extend(validate_chunk_plan_schema(results.get("chunk_plan", {})))
    for key in ["selected_models", "unsupported_models", "errors"]:
        _require_list(results, key, errors, "results")
    runtime_rankings = _require_object(results, "runtime_rankings", errors, "results")
    if runtime_rankings and runtime_rankings.get("schema") != "easy_asr_bench.runtime_rankings.v1":
        errors.append("runtime_rankings.schema must be easy_asr_bench.runtime_rankings.v1")
    plan_chunk_ids = {str(chunk.get("chunk_id")) for chunk in results.get("chunk_plan", {}).get("chunks", []) if isinstance(chunk, dict)}
    runs = _require_list(results, "runs", errors, "results")
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            errors.append(f"run {index} must be an object")
            continue
        for key in ["model", "transcript_chunks", "metrics", "errors"]:
            if key not in run:
                errors.append(f"run {index} missing {key}")
        model = run.get("model", {})
        if isinstance(model, dict):
            _require_keys(model, ["candidate_id", "display_name", "adapter_name"], errors, f"run {index} model")
        else:
            errors.append(f"run {index} model must be an object")
        transcript_chunks = run.get("transcript_chunks", [])
        if not isinstance(transcript_chunks, list):
            errors.append(f"run {index} transcript_chunks must be a list")
            transcript_chunks = []
        for chunk_index, chunk in enumerate(transcript_chunks):
            path = f"run {index} transcript_chunks[{chunk_index}]"
            if not isinstance(chunk, dict):
                errors.append(f"{path} must be an object")
                continue
            _require_keys(chunk, ["chunk_id", "start_seconds", "end_seconds", "text"], errors, path)
            if not isinstance(chunk.get("text", ""), str):
                errors.append(f"{path}.text must be a string")
            chunk_id = str(chunk.get("chunk_id", ""))
            if plan_chunk_ids and chunk_id and chunk_id not in plan_chunk_ids:
                errors.append(f"{path}.chunk_id {chunk_id} is not in chunk_plan")
        metrics = run.get("metrics", {})
        if isinstance(metrics, dict):
            for key in ["provider", "peak_process_memory_mb", "peak_vram_mb", "vram_measurement_source"]:
                if key not in metrics:
                    errors.append(f"run {index} metrics missing {key}")
            if "provider" in metrics and not isinstance(metrics["provider"], str):
                errors.append(f"run {index} metrics provider must be a string")
            if "peak_process_memory_mb" in metrics and not _is_number(metrics["peak_process_memory_mb"]):
                errors.append(f"run {index} metrics peak_process_memory_mb must be a number")
        else:
            errors.append(f"run {index} metrics must be an object")
        run_errors = run.get("errors", [])
        if not isinstance(run_errors, list):
            errors.append(f"run {index} errors must be a list")
            run_errors = []
        for error_index, error in enumerate(run_errors):
            errors.extend(validate_run_error_schema(error, f"run {index} errors[{error_index}]"))
    for error_index, error in enumerate(results.get("errors", []) if isinstance(results.get("errors", []), list) else []):
        errors.extend(validate_run_error_schema(error, f"results.errors[{error_index}]"))
    return errors
