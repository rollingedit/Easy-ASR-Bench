from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.adapters.base import ChunkTranscript, ModelCandidate, ModelRunResult
from app.results_writer import build_results, write_all_reports
from qa.runtime_matrix.common import write_row


def _candidate(candidate_id: str, display_name: str, *, groups: list[str] | None = None) -> ModelCandidate:
    return ModelCandidate(
        candidate_id=candidate_id,
        display_name=display_name,
        family_name=display_name,
        backend="fixture",
        container_format="fixture",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="FP32",
        path=Path("Models") / candidate_id,
        adapter_name="fixture",
        runnable=True,
        dependency_groups=groups or [],
        metadata={"groups": groups or []},
    )


def _chunk(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        start_seconds=float(index),
        end_seconds=float(index + 1),
        cut_reason="fixture",
        rms_db=-20.0,
    )


def _run(candidate: ModelCandidate, chunks: list[SimpleNamespace], texts: list[str], errors: list[str | dict] | None = None) -> ModelRunResult:
    return ModelRunResult(
        candidate=candidate,
        transcript_chunks=[
            ChunkTranscript(
                chunk_id=f"{chunk.index + 1:04d}",
                start_seconds=chunk.start_seconds,
                end_seconds=chunk.end_seconds,
                text=texts[index],
                raw={"fixture": True},
            )
            for index, chunk in enumerate(chunks)
        ],
        metrics={
            "provider": "fixture",
            "audio_seconds": float(len(chunks)),
            "model_load_seconds": 0.01,
            "inference_seconds": 0.05,
            "total_wall_seconds": 0.1,
            "audio_seconds_per_wall_second": 5.0,
            "peak_process_memory_mb": 80.0,
            "peak_vram_mb": None,
            "vram_measurement_source": "unavailable",
        },
        errors=errors or [],
    )


def _write_failure_report(row_id: str, evidence_dir: Path) -> dict:
    source = evidence_dir / "Input" / "fixture.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"failure-isolation-fixture")
    chunks = [_chunk(0), _chunk(1)]
    good = _candidate("good_model", "Good model")
    bad = _candidate("bad_model", "Bad model")
    good_run = _run(good, chunks, ["good first chunk", "good second chunk"])
    model_error = {
        "status": "model_failed",
        "stage": "model_load",
        "model_id": "bad_model",
        "model_name": "Bad model",
        "message": "fixture model load failed",
        "likely_causes": ["fixture dependency/provider failure"],
        "next_actions": ["skip affected model and continue remaining models"],
        "repair_command": "Run setup.bat --doctor for runtime status and repair commands.",
        "log_path": str(evidence_dir / "Logs" / "bad_model.log"),
    }
    chunk_error = "0002: fixture chunk decode failed"
    if row_id == "one_model_failure_continues":
        run_results = [good_run, _run(bad, chunks, ["", ""], [model_error])]
    elif row_id == "one_chunk_failure_continues":
        run_results = [_run(good, chunks, ["good first chunk", ""], [chunk_error])]
    else:
        run_results = [good_run, _run(bad, chunks, ["", ""], [model_error]), _run(_candidate("chunky", "Chunk failure model"), chunks, ["ok", ""], [chunk_error])]
    results = build_results(
        source,
        audio_seconds=2.0,
        chunks=chunks,
        run_results=run_results,
        unsupported_models=[],
        media_seconds=0.01,
    )
    output_dir = write_all_reports(results, evidence_dir / "Output")
    json_data = (output_dir / "results.json").read_text(encoding="utf-8")
    txt_data = (output_dir / "results.txt").read_text(encoding="utf-8")
    html_data = (output_dir / "compare.html").read_text(encoding="utf-8")
    failures: list[str] = []
    if "good first chunk" not in json_data + txt_data + html_data:
        failures.append("successful transcript chunks missing from report artifacts")
    if row_id in {"one_model_failure_continues", "batch_continues_after_one_model_or_chunk_fails"} and "fixture model load failed" not in json_data + txt_data + html_data:
        failures.append("model failure was not persisted in reports")
    if row_id in {"one_chunk_failure_continues", "batch_continues_after_one_model_or_chunk_fails"} and "chunk_inference" not in json_data + txt_data + html_data:
        failures.append("chunk failure was not normalized and persisted in reports")
    if not (output_dir / "benchmark.csv").exists():
        failures.append("benchmark.csv missing")
    details = {
        "output_dir": str(output_dir),
        "run_count": len(run_results),
        "chunk_count": len(chunks),
        "errors_by_model": {run.candidate.candidate_id: len(run.errors) for run in run_results},
        "failures": failures,
    }
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "Batch/report path preserved successful work while persisting model or chunk failure details."
            if not failures
            else "Failure-isolation report validation failed."
        ),
        details=details,
        artifacts=[output_dir / "results.json", output_dir / "results.txt", output_dir / "benchmark.csv", output_dir / "compare.html"],
    )


class _FakeAdapter:
    name = "fixture"

    def required_dependency_groups(self, candidate: ModelCandidate) -> list[str]:
        return list(candidate.metadata.get("groups", []))


def _dependency_decline(row_id: str, evidence_dir: Path) -> dict:
    import app.main as main

    good = _candidate("good_model", "Good model")
    bad = _candidate("needs_onnx", "Needs ONNX", groups=["onnx"])
    original_adapter_for = main.adapter_for
    import app.dependency_manager as dependency_manager

    original_missing = dependency_manager.missing_modules_for_config
    try:
        main.adapter_for = lambda candidate: _FakeAdapter()
        dependency_manager.missing_modules_for_config = lambda group, config: ["onnxruntime"] if group == "onnx" else []
        kept, kept_llm = main.ensure_dependencies(
            [good, bad],
            {"dependency_install": {"auto_install_missing_runtime_dependencies": False}},
        )
    finally:
        main.adapter_for = original_adapter_for
        dependency_manager.missing_modules_for_config = original_missing
    kept_ids = [candidate.candidate_id for candidate in kept]
    failures = []
    if kept_ids != ["good_model"]:
        failures.append(f"expected only good_model to remain, got {kept_ids}")
    if kept_llm is not None:
        failures.append("unexpected reference LLM returned")
    details = {
        "kept_model_ids": kept_ids,
        "skipped_model_ids": ["needs_onnx"] if "needs_onnx" not in kept_ids else [],
        "missing_dependency_group": "onnx",
        "missing_modules": ["onnxruntime"],
        "auto_install_missing_runtime_dependencies": False,
        "failures": failures,
    }
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "Declined dependency install skips only affected models while preserving runnable models."
            if not failures
            else "Dependency-decline isolation validation failed."
        ),
        details=details,
    )


def _dependency_accept(row_id: str, evidence_dir: Path) -> dict:
    import app.main as main
    import app.dependency_manager as dependency_manager

    good = _candidate("good_model", "Good model")
    repaired = _candidate("needs_onnx", "Needs ONNX", groups=["onnx"])
    installed_groups: list[str] = []
    install_log_paths: list[str] = []
    confirmation_prompts: list[dict[str, str]] = []

    original_adapter_for = main.adapter_for
    original_confirmation = main._dependency_install_confirmation
    original_batch_confirmation = main._dependency_install_batch_confirmation
    original_missing = dependency_manager.missing_modules_for_config
    original_install = dependency_manager.install_group_for_config
    try:
        main.adapter_for = lambda candidate: _FakeAdapter()
        main._dependency_install_confirmation = lambda group, command: (
            confirmation_prompts.append({"group": group, "command": command}) or "install"
        )
        main._dependency_install_batch_confirmation = lambda groups, commands=None: (
            confirmation_prompts.extend({"group": group, "command": (commands or {}).get(group, "")} for group in groups) or "install"
        )

        def fake_missing(group: str, config: dict) -> list[str]:
            if group == "onnx" and group not in installed_groups:
                return ["onnxruntime"]
            return []

        def fake_install(group: str, root: Path, config: dict, log_path: Path | None = None) -> dict:
            installed_groups.append(group)
            if log_path is not None:
                install_log_paths.append(str(log_path))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("fixture dependency repair accepted\n", encoding="utf-8")
            return {"group": group, "fixture_repair": True}

        dependency_manager.missing_modules_for_config = fake_missing
        dependency_manager.install_group_for_config = fake_install
        kept, kept_llm = main.ensure_dependencies(
            [good, repaired],
            {
                "folders": {"logs": str(evidence_dir / "Logs")},
                "dependency_install": {"auto_install_missing_runtime_dependencies": True},
            },
        )
    finally:
        main.adapter_for = original_adapter_for
        main._dependency_install_confirmation = original_confirmation
        main._dependency_install_batch_confirmation = original_batch_confirmation
        dependency_manager.missing_modules_for_config = original_missing
        dependency_manager.install_group_for_config = original_install

    kept_ids = [candidate.candidate_id for candidate in kept]
    failures = []
    if kept_ids != ["good_model", "needs_onnx"]:
        failures.append(f"expected both models to remain after accepted repair, got {kept_ids}")
    if installed_groups != ["onnx"]:
        failures.append(f"expected exactly one onnx repair attempt, got {installed_groups}")
    if not confirmation_prompts:
        failures.append("dependency install confirmation was not requested")
    if kept_llm is not None:
        failures.append("unexpected reference LLM returned")
    if install_log_paths and not Path(install_log_paths[0]).exists():
        failures.append("dependency install log path was not written")
    details = {
        "kept_model_ids": kept_ids,
        "installed_groups": installed_groups,
        "missing_before": {"onnx": ["onnxruntime"]},
        "missing_after": {"onnx": [] if "onnx" in installed_groups else ["onnxruntime"]},
        "confirmation_prompts": confirmation_prompts,
        "install_log_paths": install_log_paths,
        "auto_install_missing_runtime_dependencies": True,
        "failures": failures,
    }
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "Accepted dependency repair runs the product install path and keeps repaired plus already-runnable models."
            if not failures
            else "Dependency-accept repair validation failed."
        ),
        details=details,
        artifacts=[Path(path) for path in install_log_paths],
    )


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id == "dependency_install_declined":
        return _dependency_decline(row_id, evidence_dir)
    if row_id == "dependency_install_accepted":
        return _dependency_accept(row_id, evidence_dir)
    if row_id in {"batch_continues_after_one_model_or_chunk_fails", "one_model_failure_continues", "one_chunk_failure_continues"}:
        return _write_failure_report(row_id, evidence_dir)
    return write_row(
        row_id,
        "fail",
        evidence_dir,
        summary=f"Unsupported failure-isolation row id: {row_id}",
        details={"row_id": row_id},
    )
