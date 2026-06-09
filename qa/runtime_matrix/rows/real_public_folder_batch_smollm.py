from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.batch_report import write_batch_report
from app.config import load_config
from app.html_report_builder import build_html_report
from app.main import process_file_with_candidates
from app.model_scanner import scan_models
from app.reference_import import import_llm_reference
from qa.runtime_matrix.common import dependency_resolution_report_failures, package_versions, write_row
from qa.runtime_matrix.rows.generic_onnx_smollm_grading import _reference_for_public_media
from qa.runtime_matrix.rows.real_public_media_faster_whisper_smollm import _download_fixture
from qa.runtime_matrix.rows.same_media_multi_model_smollm_benchmark import GROUPS
from qa.runtime_matrix.rows.same_media_multi_model_smollm_benchmark import _ensure_faster_whisper
from qa.runtime_matrix.rows.same_media_multi_model_smollm_benchmark import _ensure_generic_onnx_quality
from qa.runtime_matrix.rows.same_media_multi_model_smollm_benchmark import _repair_dependencies
from qa.runtime_matrix.rows.smollm_reference_grading_report import SMOLLM_PATH, _smollm_candidate


MIN_PUBLIC_MEDIA_SECONDS = 20.0
LONG_PUBLIC_MEDIA_REFERENCES = {
    "wikimedia_public_domain_gettysburg_ogg": (
        "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, "
        "and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, "
        "testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great "
        "battlefield of that war. We have come to dedicate a portion of that field as a final resting place for those who "
        "here gave their lives that this nation might live. It is altogether fitting and proper that we should do this. "
        "But in a larger sense, we cannot dedicate, we cannot consecrate, we cannot hallow this ground. The brave men, "
        "living and dead, who struggled here, have consecrated it far above our poor power to add or detract. The world "
        "will little note nor long remember what we say here, but it can never forget what they did here. It is for us, "
        "the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly "
        "advanced. It is rather for us to be here dedicated to the great task remaining before us, that from these honored "
        "dead we take increased devotion to that cause for which they gave the last full measure of devotion, that we here "
        "highly resolve that these dead shall not have died in vain, that this nation, under God, shall have a new birth "
        "of freedom, and that government of the people, by the people, for the people, shall not perish from the earth."
    ),
    "wikimedia_public_domain_spoken_words_webm": (
        "They said HIV was always done. It was her, her, her, her, sometimes him. "
        "But never did I foresee the face of HIV being me. Know your HIV status. Get tested."
    ),
}


def _score_report(output_dir: Path, expected_text: str) -> tuple[dict, Path]:
    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    reference = _reference_for_public_media(results, expected_text)
    scored = import_llm_reference(results, "SmolLM corrected reference fixture:\n```json\n" + json.dumps(reference) + "\n```")
    (output_dir / "scored_report.json").write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    scored_html = output_dir / "compare_scored.html"
    scored_html.write_text(build_html_report(scored), encoding="utf-8", newline="\n")
    return scored, scored_html


def run(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    if row_id != "real_public_folder_batch_smollm_benchmark":
        return write_row(row_id, "fail", evidence_dir, summary=f"Unhandled batch row: {row_id}")
    if not SMOLLM_PATH.exists():
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="SmolLM 135M GGUF fixture is missing, so the folder batch cannot prove local LLM correction.",
            block_reason=f"missing {SMOLLM_PATH}",
            external_requirement="download HuggingFaceTB/SmolLM-135M-GGUF Q4_K_M fixture",
        )

    config = load_config(Path("config.json"))
    config["folders"] = {
        "models": str(evidence_dir / "Models"),
        "input": str(evidence_dir / "Input"),
        "output": str(evidence_dir / "Output"),
        "temp": str(evidence_dir / "Temp"),
        "logs": str(evidence_dir / "Logs"),
        "cache": str(evidence_dir / "Cache"),
    }
    config["runtime"]["provider"] = "cpu"
    config["runtime"]["prefer_gpu"] = False
    config["runtime"]["llm_context_tokens"] = 1024
    config["runtime"]["llm_reference_max_tokens"] = 128
    config["runtime"]["llm_reference_temperature"] = 0.0
    config["input"]["recursive_folders"] = True

    blockers, dependency_details, dependency_artifacts = _repair_dependencies(config, evidence_dir, install_deps)
    if blockers:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="One or more dependency groups are not runnable for the real public folder batch.",
            block_reason="; ".join(blockers),
            external_requirement="rerun with --install-deps or repair the listed dependency groups through setup.bat",
            details={**dependency_details, "dependency_versions": package_versions(["faster-whisper", "ctranslate2", "onnxruntime", "llama-cpp-python"])},
            artifacts=dependency_artifacts,
        )

    models_root = Path(config["folders"]["models"])
    fixture_errors: list[str] = []
    for path, error in [
        _ensure_faster_whisper(models_root, allow_downloads),
        _ensure_generic_onnx_quality(models_root, allow_downloads)[:2],
    ]:
        if error:
            fixture_errors.append(error)
    media_rows = []
    input_dir = Path(config["folders"]["input"]) / "public_media_batch"
    input_dir.mkdir(parents=True, exist_ok=True)
    for fixture_id in ["wikimedia_public_domain_gettysburg_ogg", "wikimedia_public_domain_spoken_words_webm"]:
        source, details, error = _download_fixture(fixture_id, evidence_dir, allow_downloads)
        if error:
            fixture_errors.append(error)
            continue
        destination = input_dir / source.name
        shutil.copy2(source, destination)
        media_rows.append((fixture_id, destination, LONG_PUBLIC_MEDIA_REFERENCES[fixture_id]))
    if fixture_errors:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="One or more model or media fixtures are not staged for the real public folder batch.",
            block_reason="; ".join(fixture_errors),
            external_requirement="rerun with --allow-downloads or stage the listed cached fixtures",
            details=dependency_details,
            artifacts=dependency_artifacts,
        )

    runnable, unsupported = scan_models(models_root)
    selected = []
    for adapter_name in ["faster_whisper", "generic_onnx_manifest"]:
        matches = [candidate for candidate in runnable if candidate.adapter_name == adapter_name]
        if matches:
            selected.append(matches[0])
    llm_candidate, scan_details = _smollm_candidate()
    failures: list[str] = []
    if len(selected) < 2:
        failures.append("folder batch did not discover both faster-whisper and Generic ONNX ASR candidates")
    if llm_candidate is None:
        failures.append("SmolLM GGUF was not classified as a reference/correction LLM candidate")
    if failures:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="Real public folder batch preflight failed.",
            details={**dependency_details, **scan_details, "selected_adapters": [candidate.adapter_name for candidate in selected], "failures": failures},
            artifacts=dependency_artifacts,
        )

    batch_rows = []
    scored_paths: list[Path] = []
    score_statuses = []
    durations = {}
    for fixture_id, source, expected_text in media_rows:
        output_dir = process_file_with_candidates(source, selected, config, unsupported, reference_llm=llm_candidate)
        status = "done" if output_dir else "failed"
        batch_rows.append({"source_path": str(source.resolve()), "status": status, "output_path": str(output_dir or "")})
        if output_dir is None:
            failures.append(f"{source.name} did not produce a report directory")
            continue
        scored, scored_html = _score_report(output_dir, expected_text)
        scored_paths.append(scored_html)
        score_statuses.append(scored.get("status"))
        results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        duration = float(results.get("source", {}).get("duration_seconds") or 0.0)
        durations[fixture_id] = duration
        if duration < MIN_PUBLIC_MEDIA_SECONDS:
            failures.append(f"{source.name} was only {duration:.3f}s; real public folder batch samples must be at least {MIN_PUBLIC_MEDIA_SECONDS:.1f}s")
        dependency_failures, _details = dependency_resolution_report_failures(results, expected_groups={"faster_whisper", "onnx", "llama_cpp"})
        failures.extend(f"{source.name}: {failure}" for failure in dependency_failures)
        if len(results.get("runs", [])) != len(selected):
            failures.append(f"{source.name} did not run every selected ASR model")
        if scored.get("status") != "scored":
            failures.append(f"{source.name} did not produce scored LLM reference output")
        if "Loaded LLM-Corrected Reference" not in scored_html.read_text(encoding="utf-8"):
            failures.append(f"{source.name} scored HTML does not show the loaded corrected reference")

    batch_dir = write_batch_report(Path(config["folders"]["output"]), batch_rows)
    batch_html = (batch_dir / "final_results.html").read_text(encoding="utf-8")
    if len(batch_rows) != 2 or "wikimedia_public_domain_gettysburg_ogg" not in batch_html or "wikimedia_public_domain_spoken_words_webm" not in batch_html:
        failures.append("batch dashboard does not include both long real public media files")
    if "Corrected reference" not in batch_html or "Model transcripts" not in batch_html:
        failures.append("batch dashboard does not present corrected references and model transcripts")
    if "Open report" not in batch_html:
        failures.append("batch dashboard does not link per-file reports")
    visible_names = sorted(path.name for path in batch_dir.iterdir() if path.is_file())
    if visible_names != ["final_results.html"]:
        failures.append(f"batch dashboard top-level files are cluttered: {visible_names}")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "Real public audio/video folder batch ran multiple ASR models per file, then SmolLM scoring and the batch dashboard completed."
            if not failures
            else "Real public audio/video folder batch validation failed."
        ),
        details={
            **dependency_details,
            **scan_details,
            "selected_adapters": [candidate.adapter_name for candidate in selected],
            "file_count": len(batch_rows),
            "minimum_media_seconds": MIN_PUBLIC_MEDIA_SECONDS,
            "durations": durations,
            "score_statuses": score_statuses,
            "batch_dir": str(batch_dir),
            "failures": failures,
        },
        artifacts=[
            *dependency_artifacts,
            input_dir,
            batch_dir / "_data" / "batch.json",
            batch_dir / "_data" / "batch-records.json",
            batch_dir / "final_results.html",
            *[Path(row["output_path"]) / "results.json" for row in batch_rows if row.get("output_path")],
            *[Path(row["output_path"]) / "compare.html" for row in batch_rows if row.get("output_path")],
            *scored_paths,
        ],
    )
