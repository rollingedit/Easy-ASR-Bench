from __future__ import annotations

import json
from pathlib import Path

from app.adapters.gguf_llm_reference import GGUFLLMReferenceAdapter
from app.html_report_builder import build_html_report
from app.reference_import import import_llm_reference
from app.results_writer import render_text_report, write_benchmark_csv
from qa.run_real_tiny_model_smoke import REFERENCE_TEXT
from qa.runtime_matrix.common import dependency_resolution_report_failures, package_versions, write_row
from qa.runtime_matrix.rows.real_tiny_faster_whisper_report_smoke import _latest_smoke_payload, run as run_faster_whisper_smoke
from qa.runtime_matrix.rows.report_reference_validation import _assert_report_files
from qa.runtime_matrix.rows.smollm_reference_grading_report import SMOLLM_PATH, _smollm_candidate


def _reference_for_real_smoke(results: dict) -> dict:
    segments = []
    for index, chunk in enumerate(results.get("chunk_plan", {}).get("chunks", [])):
        segments.append(
            {
                "chunk_id": chunk["chunk_id"],
                "start_seconds": chunk["start_seconds"],
                "end_seconds": chunk["end_seconds"],
                "text": REFERENCE_TEXT if index == 0 else "",
                "uncertain": [],
            }
        )
    return {
        "schema": "easy_asr_bench.llm_reference.v1",
        "source_sha256": results["source"]["sha256"],
        "reference_type": "llm_corrected_reference",
        "segments": segments,
        "global_notes": [
            "Reference text comes from the deterministic Windows SAPI phrase used by the real faster-whisper smoke row."
        ],
    }


def _rewrite_base_reports(output_dir: Path, results: dict) -> None:
    (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output_dir / "results.txt").write_text(render_text_report(results), encoding="utf-8", newline="\n")
    (output_dir / "compare.html").write_text(build_html_report(results), encoding="utf-8", newline="\n")
    write_benchmark_csv(output_dir / "benchmark.csv", results)


def run(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    if row_id != "real_tiny_faster_whisper_smollm_grading":
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary=f"Unsupported real faster-whisper SmolLM grading row id: {row_id}",
            details={"row_id": row_id},
        )
    if not SMOLLM_PATH.exists():
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="SmolLM 135M GGUF fixture is not present locally, so real faster-whisper output cannot be graded by the local reference path.",
            block_reason=f"missing {SMOLLM_PATH}",
            external_requirement="download HuggingFaceTB/SmolLM-135M-GGUF Q4_K_M fixture",
        )
    try:
        import llama_cpp  # noqa: F401
    except ModuleNotFoundError:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="llama-cpp-python is not installed, so SmolLM GGUF cannot run after real faster-whisper ASR.",
            block_reason="missing llama_cpp import",
            external_requirement="install llama_cpp dependency group",
            details={"dependency_versions": package_versions(["llama-cpp-python"])},
            artifacts=[SMOLLM_PATH],
        )

    smoke_row = run_faster_whisper_smoke("real_tiny_faster_whisper_report_smoke", evidence_dir / "faster_whisper_smoke", install_deps, allow_downloads)
    if smoke_row.get("status") != "pass":
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Real faster-whisper smoke did not pass, so post-ASR SmolLM grading cannot be trusted.",
            block_reason=smoke_row.get("summary", "real faster-whisper smoke did not pass"),
            external_requirement="repair faster_whisper/CTranslate2 runtime and rerun with --install-deps if needed",
            details={"smoke_row": smoke_row},
            artifacts=[SMOLLM_PATH],
        )

    payload, payload_path = _latest_smoke_payload()
    output_dir = Path(payload["report_dir"])
    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    candidate, scan_details = _smollm_candidate()
    if candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference/correction LLM candidate.",
            details=scan_details,
            artifacts=[SMOLLM_PATH, payload_path],
        )

    adapter = GGUFLLMReferenceAdapter()
    try:
        llm = adapter.load(candidate, {"provider": "cpu", "prefer_gpu": False, "llm_context_tokens": 512})
        response = llm("Answer with the word pass.", max_tokens=8, temperature=0.0)
    except Exception as exc:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF failed to load or generate after real faster-whisper ASR output.",
            details={"error": str(exc), "dependency_versions": package_versions(["llama-cpp-python"]), **scan_details},
            artifacts=[SMOLLM_PATH, payload_path, results_path],
        )
    generated_text = response["choices"][0]["text"].strip() if isinstance(response, dict) else str(response).strip()
    if not generated_text:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF loaded after real faster-whisper ASR but generated empty text.",
            details={"dependency_versions": package_versions(["llama-cpp-python"])},
            artifacts=[SMOLLM_PATH, payload_path, results_path],
        )

    results["reference_llm"] = {
        "candidate_id": candidate.candidate_id,
        "display_name": candidate.display_name,
        "path": str(candidate.path),
    }
    results["local_llm_reference_attempt"] = {
        "candidate_id": candidate.candidate_id,
        "display_name": candidate.display_name,
        "status": "generated",
        "raw_response": generated_text,
        "note": "This row proves SmolLM runs after real faster-whisper ASR. Stable scoring uses the known SAPI reference phrase.",
    }
    _rewrite_base_reports(output_dir, results)

    reference = _reference_for_real_smoke(results)
    scored = import_llm_reference(results, "SmolLM corrected reference fixture:\n```json\n" + json.dumps(reference) + "\n```")
    scored_path = output_dir / "scored_report.json"
    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    scored_results = dict(results)
    if scored.get("status") == "scored":
        scored_results["reference_scores"] = scored["scores"]
    scored_html = output_dir / "compare_scored.html"
    scored_html.write_text(build_html_report(scored_results), encoding="utf-8", newline="\n")

    failures = [
        failure
        for failure in _assert_report_files(output_dir, large=False)
        if failure != "compare_scored.html missing marker fixture_windows_gpu_adapter_memory"
    ]
    dependency_report_failures, dependency_report_details = dependency_resolution_report_failures(results, expected_groups={"faster_whisper", "llama_cpp"})
    failures.extend(dependency_report_failures)
    report_text = (output_dir / "results.txt").read_text(encoding="utf-8")
    scored_html_text = scored_html.read_text(encoding="utf-8")
    if "windows_gpu_adapter_memory" not in scored_html_text:
        failures.append("compare_scored.html missing real Windows GPU Adapter Memory VRAM marker")
    if "Local GGUF Reference/Correction LLM" not in report_text:
        failures.append("results.txt missing local GGUF reference/correction LLM section")
    if generated_text not in report_text:
        failures.append("results.txt missing SmolLM generated text")
    if scored.get("status") != "scored":
        failures.append("real faster-whisper smoke reference import did not score")
    run_id = results["runs"][0]["model"]["candidate_id"]
    score = scored.get("scores", {}).get(run_id, {})
    if score.get("normalized_wer") is None:
        failures.append("scored report missing real faster-whisper normalized WER")
    if float(score.get("normalized_wer", 1.0)) > float(results["runs"][0]["metrics"].get("real_smoke_max_normalized_wer", 0.60)):
        failures.append("scored WER exceeded real smoke threshold")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "Real faster-whisper ASR output was followed by SmolLM GGUF generation and scored report validation."
            if not failures
            else "Real faster-whisper SmolLM grading validation failed."
        ),
        details={
            "smoke_payload": payload,
            "output_dir": str(output_dir),
            "candidate_id": candidate.candidate_id,
            "generated_text": generated_text,
            "score_status": scored.get("status"),
            "real_faster_whisper_score": {
                "candidate_id": run_id,
                "normalized_wer": score.get("normalized_wer"),
                "balanced_score": score.get("balanced_score"),
                "balanced_rank": score.get("balanced_rank"),
                "alignment_mode": score.get("alignment_mode"),
            },
            "dependency_versions": package_versions(["faster-whisper", "ctranslate2", "setuptools", "llama-cpp-python"]),
            "failures": failures,
            **dependency_report_details,
            **scan_details,
        },
        artifacts=[
            SMOLLM_PATH,
            payload_path,
            output_dir / "results.json",
            output_dir / "results.txt",
            output_dir / "benchmark.csv",
            output_dir / "compare.html",
            scored_path,
            scored_html,
        ],
    )
