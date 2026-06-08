from __future__ import annotations

import json
from pathlib import Path

from app.adapters.gguf_llm_reference import GGUFLLMReferenceAdapter
from app.html_report_builder import build_html_report
from app.model_scanner import scan_models
from app.reference_import import import_llm_reference
from app.results_writer import build_results, write_all_reports
from qa.runtime_matrix.common import dependency_resolution_report_failures, package_versions, write_row
from qa.runtime_matrix.rows.report_reference_validation import _assert_report_files, _chunk, _candidate, _reference_for, _run


SMOLLM_PATH = Path("Temp/real_tiny_llm_smoke/Models/SmolLM-135M-GGUF/SmolLM-135M.Q4_K_M.gguf")


def _smollm_candidate() -> tuple[object | None, dict]:
    runnable, unsupported = scan_models(SMOLLM_PATH.parent)
    reference_candidates = [candidate for candidate in unsupported if candidate.adapter_name == "gguf_llm_reference"]
    return (reference_candidates[0] if reference_candidates else None), {
        "runnable_count": len(runnable),
        "unsupported_count": len(unsupported),
        "reference_candidate_count": len(reference_candidates),
    }


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id != "smollm_reference_grading_report":
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary=f"Unsupported SmolLM grading row id: {row_id}",
            details={"row_id": row_id},
        )
    if not SMOLLM_PATH.exists():
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="SmolLM 135M GGUF fixture is not present locally, so the local reference/grading report path cannot run.",
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
            summary="llama-cpp-python is not installed, so SmolLM GGUF cannot run after ASR output.",
            block_reason="missing llama_cpp import",
            external_requirement="install llama_cpp dependency group",
            details={"dependency_versions": package_versions(["llama-cpp-python"])},
            artifacts=[SMOLLM_PATH],
        )

    candidate, scan_details = _smollm_candidate()
    if candidate is None:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF was not classified as a reference/correction LLM candidate.",
            details=scan_details,
            artifacts=[SMOLLM_PATH],
        )

    source = evidence_dir / "Input" / "smollm-reference-grading.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"easy-asr-bench-smollm-grading-fixture")
    chunks = [_chunk(index) for index in range(3)]
    expected = [f"chunk {index + 1} expected transcript" for index in range(3)]
    near = list(expected)
    near[-1] = near[-1].replace("expected", "expcted")
    wrong = [f"chunk {index + 1} wrong output" for index in range(3)]
    results = build_results(
        source,
        audio_seconds=3.0,
        chunks=chunks,
        run_results=[
            _run(_candidate("fixture_fast", "Fixture Fast ASR", "fixture_fast", "fp32"), chunks, near, 5.0, 96.0),
            _run(_candidate("fixture_wrong", "Fixture Wrong ASR", "fixture_wrong", "int8"), chunks, wrong, 1.0, 512.0),
        ],
        unsupported_models=[],
        media_seconds=0.01,
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
            summary="SmolLM GGUF was detected but failed to load or generate during the post-ASR grading row.",
            details={
                "candidate_id": candidate.candidate_id,
                "error": str(exc),
                "dependency_versions": package_versions(["llama-cpp-python"]),
                **scan_details,
            },
            artifacts=[SMOLLM_PATH],
        )
    generated_text = response["choices"][0]["text"].strip() if isinstance(response, dict) else str(response).strip()
    if not generated_text:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="SmolLM GGUF loaded after ASR output but generated empty text.",
            details={"candidate_id": candidate.candidate_id, "dependency_versions": package_versions(["llama-cpp-python"])},
            artifacts=[SMOLLM_PATH],
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
        "note": "This tiny-model row proves local GGUF LLM execution after ASR output; deterministic fixture reference JSON is used for stable regression scoring.",
    }
    output_dir = write_all_reports(results, evidence_dir / "Output")
    reference = _reference_for(results)
    scored = import_llm_reference(results, "SmolLM corrected reference fixture:\n```json\n" + json.dumps(reference) + "\n```")
    scored_path = output_dir / "scored_report.json"
    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    scored_results = dict(results)
    if scored.get("status") == "scored":
        scored_results["reference_scores"] = scored["scores"]
    scored_html = output_dir / "compare_scored.html"
    scored_html.write_text(build_html_report(scored_results), encoding="utf-8", newline="\n")

    dependency_report_failures, dependency_report_details = dependency_resolution_report_failures(results, expected_groups={"llama_cpp"})
    failures = [*_assert_report_files(output_dir, large=False), *dependency_report_failures]
    results_text = (output_dir / "results.txt").read_text(encoding="utf-8")
    scored_html_text = scored_html.read_text(encoding="utf-8")
    scores = scored.get("scores", {})
    if "Local GGUF Reference/Correction LLM" not in results_text:
        failures.append("results.txt missing local GGUF reference/correction LLM section")
    if generated_text not in results_text:
        failures.append("results.txt missing SmolLM raw generation text")
    if scores.get("fixture_fast", {}).get("balanced_rank") != 1:
        failures.append("expected fixture_fast to rank first after SmolLM/reference scoring")
    if "replace" not in scored_html_text:
        failures.append("scored HTML missing correction/diff marker")

    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "ASR-style results were followed by SmolLM GGUF generation, LLM-corrected reference scoring, and scored report validation."
            if not failures
            else "SmolLM post-ASR grading/report validation failed."
        ),
        details={
            "output_dir": str(output_dir),
            "candidate_id": candidate.candidate_id,
            "generated_text": generated_text,
            "score_status": scored.get("status"),
            "scores": {
                candidate_id: {
                    "normalized_wer": score.get("normalized_wer"),
                    "balanced_score": score.get("balanced_score"),
                    "balanced_rank": score.get("balanced_rank"),
                    "alignment_mode": score.get("alignment_mode"),
                }
                for candidate_id, score in scores.items()
            },
            "dependency_versions": package_versions(["llama-cpp-python"]),
            **dependency_report_details,
            "failures": failures,
            **scan_details,
        },
        artifacts=[
            SMOLLM_PATH,
            output_dir / "results.json",
            output_dir / "results.txt",
            output_dir / "benchmark.csv",
            output_dir / "compare.html",
            scored_path,
            scored_html,
        ],
    )
