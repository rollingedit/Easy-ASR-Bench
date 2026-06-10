import json
from pathlib import Path

from app.batch_report import render_batch_html, render_results_index_html, write_batch_report
from qa.runtime_matrix.rows.real_public_folder_batch_smollm import LONG_PUBLIC_MEDIA_REFERENCES, MIN_PUBLIC_MEDIA_SECONDS


def make_result(folder: Path, model_name: str, model_count: int = 1, with_scored: bool = False) -> None:
    folder.mkdir(parents=True)
    runs = []
    for index in range(model_count):
        candidate_id = f"model-{index + 1}"
        runs.append(
            {
                "model": {"candidate_id": candidate_id, "display_name": f"{model_name} {index + 1}", "backend": "fixture", "precision": "fp32"},
                "metrics": {"audio_seconds_per_wall_second": 2.5, "peak_process_memory_mb": 120, "peak_vram_mb": 1400},
                "transcript_chunks": [{"chunk_id": "0001", "text": f"{model_name} transcript"}],
                "errors": [],
            }
        )
    results = {
        "source": {"duration_seconds": 12.5},
        "chunk_plan": {"chunks": [{"chunk_id": "0001"}, {"chunk_id": "0002"}]},
        "runs": runs,
        "unsupported_models": [{"display_name": "Skipped"}],
    }
    (folder / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (folder / "compare.html").write_text("<!doctype html>", encoding="utf-8")
    if with_scored:
        scored = {
            "schema": "easy_asr_bench.scored_report.v1",
            "status": "scored",
            "results": results,
            "reference": {
                "schema": "easy_asr_bench.llm_reference.v1",
                "reference_type": "llm_corrected_reference",
                "segments": [{"chunk_id": "0001", "text": "corrected reference transcript"}],
            },
            "scores": {"model-1": {"normalized_wer": 0.125, "balanced_rank": 1}},
        }
        (folder / "scored_report.json").write_text(json.dumps(scored), encoding="utf-8")
        (folder / "compare_scored.html").write_text("<!doctype html>", encoding="utf-8")


def make_failed_result(folder: Path) -> None:
    folder.mkdir(parents=True)
    (folder / "results.json").write_text(
        json.dumps(
            {
                "source": {"duration_seconds": 0},
                "chunk_plan": {"chunks": []},
                "runs": [],
                "unsupported_models": [],
                "errors": [{"status": "failed_before_model_run", "stage": "media_probe", "message": "No audio stream was found."}],
            }
        ),
        encoding="utf-8",
    )
    (folder / "compare.html").write_text("<!doctype html>", encoding="utf-8")


def test_batch_report_does_not_count_reference_llm_as_skipped_asr_model(tmp_path: Path):
    folder = tmp_path / "out"
    make_result(folder, "Model", with_scored=True)
    results = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    results["unsupported_models"] = [
        {"display_name": "SmolLM", "category": "reference_llm"},
        {"display_name": "SmolLM old JSON", "adapter_name": "gguf_llm_reference"},
        {"display_name": "Broken ASR", "category": "asr"},
    ]
    (folder / "results.json").write_text(json.dumps(results), encoding="utf-8")

    html = render_batch_html({"files": [{"source_path": str(tmp_path / "a.wav"), "status": "done", "output_path": str(folder)}]}, tmp_path)

    assert '"unsupported_count":1' in html


def test_batch_report_renders_file_picker_transcripts_reference_and_plain_units(tmp_path: Path):
    first = tmp_path / "out1"
    second = tmp_path / "out2"
    make_result(first, "Model One", with_scored=True)
    make_result(second, "Model Two")
    payload = {
        "files": [
            {"source_path": str(tmp_path / "a.wav"), "status": "done", "output_path": str(first)},
            {"source_path": str(tmp_path / "b.mp4"), "status": "done", "output_path": str(second)},
        ]
    }

    html = render_batch_html(payload, tmp_path)

    assert "file-list" in html
    assert "selectedFile" in html
    assert "file-card" in html
    assert "Model One 1" in html
    assert "Model Two 1" in html
    assert "Model transcripts" in html
    assert "Compare model A" in html
    assert "Compare model B" in html
    assert "model-section-head" in html
    assert "compare-controls" in html
    assert "Show all" in html
    assert "compareLeftKey" in html
    assert "compareRightKey" in html
    assert "normalizeCompareKeys" in html
    assert "selectCompareLeft" in html
    assert "selectCompareRight" in html
    assert "model-grid compare-grid" in html
    assert "runs.length >= 2" in html
    assert "Reset</button>" in html
    assert '<button class="secondary-button" onclick="resetCurrentReference()">Reset</button>' in html
    assert "referenceEditStatus" in html
    assert "Edits update results automatically" in html
    assert "Results updated from pasted text" in html
    assert "reportStorageKey" in html
    assert "localStorage.setItem(reportStorageKey" in html
    assert "loadEditedReferences" in html
    assert "easy_asr_bench.batch_reference_edits.v1" in html
    assert "exportEditedReferences" in html
    assert "importEditedReferences" in html
    assert "easy-asr-bench-reference-edits.json" in html
    assert "Edits saved in this browser" in html
    assert "Restored saved edits for this report" in html
    assert "reference-actions" in html
    assert "resetCurrentReference" in html
    assert "addEventListener('keydown'" not in html
    assert "Local LLM: none used" in html
    assert "selectedModelKey" in html
    assert "modelsPerPage = 3" in html
    assert "Previous models" in html
    assert "Next models" in html
    assert "&lsaquo;" in html
    assert "&rsaquo;" in html
    assert "Models ${pageStart + 1}-${Math.min(pageStart + modelsPerPage, filteredRuns.length)} of ${filteredRuns.length}" in html
    assert "Overall model ranking" in html
    assert "aggregateModelRanking" in html
    assert "weighted by reference words" in html
    assert "Score uses word error rate weighted by corrected-reference word count" in html
    assert "File wins" in html
    assert "Total transcription time" in html
    assert "Best for this file" in html
    assert "perFileTopModels" in html
    assert "renderPerFileTopModels" in html
    assert "Model One transcript" in html
    assert "corrected reference transcript" in html
    assert "function fmtSeconds" in html
    assert "function fmtMemoryMb" in html
    assert '"duration_seconds":12.5' in html
    assert '"ram":120' in html
    assert '"vram":1400' in html
    assert "RAM peak" in html
    assert "VRAM / GPU memory peak" in html
    assert "Peak system RAM for this model run; VRAM is listed separately." in html
    assert "should not be added to it" not in html
    assert "Word error rate" in html
    assert "function wordErrorFlag" in html
    assert "Very high" in html
    assert "metric-alert" in html
    assert "Punctuation/capitalization" in html
    assert "Punctuation/case" in html
    assert "punctuation/case" in html
    assert "Readability cleanup" not in html
    assert "not scored" in html
    assert "Yellow punctuation/capitalization differences are shown for review only and do not affect the score" in html
    assert "Run completed" in html
    assert "run-status" in html
    assert "Apply pasted reference" not in html
    assert "function applyCorrectedReference" not in html
    assert "referenceUpdateTimer" in html
    assert "setTimeout(() => renderBatch(), 250)" in html
    assert "Manual LLM correction" in html
    assert "Copy prompt for this file" in html
    assert "paste it into ChatGPT or Claude" in html
    assert "paste it into ChatGPT/Claude/local LLM" not in html
    assert "copyManualPrompt" in html
    assert "markReferenceEdited" in html
    assert "updated-flash" in html
    assert "@keyframes referenceFlash" in html
    assert "void status.offsetWidth" in html
    assert "Output format:" in html
    assert "Return only the corrected transcript text" in html
    assert "Do not include a title, explanation, confidence score" in html
    assert "do not invent speaker labels" in html
    assert "Always choose the best-supported wording from the ASR outputs; do not use uncertainty placeholders." in html
    assert "[uncertain: short reason]" not in html
    assert "incomplete sentences, false starts, cut-off thoughts" in html
    assert "complete fragments into polished grammar" in html
    assert "dialogue flow, tone, speaker wording" in html
    assert "topic-specific terms, dialogue flow, names, acronyms" in html
    assert "Manual_LLM_Override_Prompt.txt" not in html
    assert "customReferenceSources" in html
    assert "Manual pasted reference" not in html
    assert "JSON is not required" not in html
    assert "renderHighlightedTranscript" in html
    assert "word-replace" in html
    assert "missing: ${escapeHtml(item.ref)}" in html
    assert "word-format" in html
    assert "Advanced details" in html
    assert "Raw batch data" in html
    assert "Open report" in html
    assert "Find the best local transcription model for these files." in html
    assert "https://github.com/rollingedit/Easy-ASR-Bench" in html


def test_write_batch_report_writes_index_and_json(tmp_path: Path):
    report_dir = write_batch_report(tmp_path, [{"source_path": "a.wav", "status": "failed", "output_path": ""}])

    assert not (report_dir / "index.html").exists()
    assert (report_dir / "final_results.html").exists()
    assert not (report_dir / "Manual_LLM_Override_Prompt.txt").exists()
    assert (report_dir / "_data" / "batch.json").exists()
    assert (report_dir / "_data" / "batch-records.json").exists()
    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "report-runs.json").exists()
    assert json.loads((report_dir / "_data" / "batch.json").read_text(encoding="utf-8"))["files"][0]["status"] == "failed"
    assert json.loads((report_dir / "_data" / "batch-records.json").read_text(encoding="utf-8"))["schema"] == "easy_asr_bench.batch_records.v1"
    report_runs = json.loads((tmp_path / "report-runs.json").read_text(encoding="utf-8"))
    assert report_runs["runs"][0]["href"].endswith("/final_results.html")
    final_html = (report_dir / "final_results.html").read_text(encoding="utf-8")
    assert "Return only the corrected transcript text" in final_html
    assert "Do not include a title, explanation, confidence score" in final_html
    assert "If speakers are not clear, do not invent speaker labels" in final_html
    assert "topic-specific terms, dialogue flow, names, acronyms" in final_html


def test_readme_documents_current_batch_output_layout():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "final_results.html" in readme
    assert "_data/" in readme
    assert "batch-records.json" in readme
    assert "Open_Latest_Report.bat`: open the newest `final_results.html`" in readme
    assert "saves pasted corrected references in the browser" in readme
    assert "  batch__20260606_143012/\n    index.html\n    batch.json" not in readme


def test_batch_report_shows_failed_file_stage_and_problem(tmp_path: Path):
    failed = tmp_path / "failed"
    make_failed_result(failed)

    html = render_batch_html(
        {"files": [{"source_path": str(tmp_path / "silent.mp4"), "status": "failed", "output_path": str(failed)}]},
        tmp_path,
    )

    assert "media_probe" in html
    assert "No audio stream was found." in html
    assert "error-summary" in html


def test_batch_report_json_scripts_are_parseable_json_not_html_entities(tmp_path: Path):
    payload = {"schema": "easy_asr_bench.batch_report.v1", "files": [{"source_path": "a&b.wav", "status": "done"}]}

    html = render_batch_html(payload, tmp_path)

    assert "&quot;" not in html
    assert '<script type="application/json" id="batch-json">{"schema"' in html
    assert "\\u0026" in html


def test_batch_report_filters_many_files_and_scrolls_transcripts(tmp_path: Path):
    files = []
    for index in range(20):
        folder = tmp_path / f"out{index}"
        make_result(folder, f"Model {index}", model_count=12)
        files.append({"source_path": str(tmp_path / f"file{index}.wav"), "status": "done", "output_path": str(folder)})

    html = render_batch_html({"files": files}, tmp_path)

    assert "Find a file or model" in html
    assert "max-height:220px" in html
    assert "selectedIndex" in html
    assert "filteredRecords" in html


def test_results_index_has_simple_dated_run_selector():
    html = render_results_index_html(
        [
            {
                "id": "batch__20260608_203936",
                "created_local": "2026-06-08T20:39:36-05:00",
                "file_count": 2,
                "completed": 2,
                "failed": 0,
                "href": "batch__20260608_203936/index.html",
            }
        ]
    )

    assert "Benchmark run" in html
    assert "Open report" in html
    assert "report_runs" in html
    assert "batch__20260608_203936/index.html" in html


def test_real_public_folder_batch_uses_long_real_audio_and_video_references():
    assert MIN_PUBLIC_MEDIA_SECONDS >= 20.0
    assert "wikimedia_cc0_word_wav" not in LONG_PUBLIC_MEDIA_REFERENCES
    assert {"wikimedia_public_domain_gettysburg_ogg", "wikimedia_public_domain_spoken_words_webm"} == set(LONG_PUBLIC_MEDIA_REFERENCES)
    assert all(len(reference.split()) >= 20 for reference in LONG_PUBLIC_MEDIA_REFERENCES.values())
