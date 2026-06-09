from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

from .html_json import json_for_script_tag
from .utils import write_json_atomic


def _safe(value: object) -> str:
    return html.escape(str(value or ""))


def write_batch_report(output_root: Path, rows: list[dict]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_root / f"batch__{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "easy_asr_bench.batch_report.v1",
        "created_local": datetime.now().astimezone().isoformat(),
        "files": rows,
    }
    records = _build_records(payload, report_dir)
    data_dir = report_dir / "_data"
    write_json_atomic(data_dir / "batch.json", payload)
    write_json_atomic(data_dir / "batch-records.json", {"schema": "easy_asr_bench.batch_records.v1", "files": records})
    (report_dir / "final_results.html").write_text(render_batch_html(payload, report_dir), encoding="utf-8", newline="\n")
    _write_results_index(output_root)
    return report_dir


def _load_result_summary(output_path: str | None) -> dict:
    if not output_path:
        return {}
    path = Path(output_path) / "results.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    scored_path = Path(output_path) / "scored_report.json"
    scores = {}
    reference_text = ""
    reference_source = ""
    if scored_path.exists():
        try:
            scored = json.loads(scored_path.read_text(encoding="utf-8"))
            scores = scored.get("scores", {}) if scored.get("status") == "scored" else {}
            reference = scored.get("reference", {})
            reference_by_chunk = {segment.get("chunk_id"): segment.get("text", "") for segment in reference.get("segments", [])}
            reference_text = "\n".join(reference_by_chunk.get(chunk.get("chunk_id"), "") for chunk in data.get("chunk_plan", {}).get("chunks", []))
            reference_llm = scored.get("results", {}).get("reference_llm") or data.get("reference_llm") or {}
            if reference_llm.get("display_name"):
                reference_source = f"Local LLM: {reference_llm.get('display_name')}"
        except (OSError, json.JSONDecodeError):
            scores = {}
            reference_text = ""
            reference_source = ""
    runs = []
    for run in data.get("runs", []):
        candidate_id = run.get("model", {}).get("candidate_id", "")
        transcript = "\n".join(chunk.get("text", "") for chunk in run.get("transcript_chunks", []))
        score = scores.get(candidate_id, {})
        runs.append(
            {
                "display_name": run.get("model", {}).get("display_name", ""),
                "candidate_id": candidate_id,
                "backend": run.get("model", {}).get("backend", ""),
                "precision": run.get("model", {}).get("precision", ""),
                "speed": run.get("metrics", {}).get("audio_seconds_per_wall_second"),
                "transcription_seconds": run.get("metrics", {}).get("total_wall_seconds"),
                "ram": run.get("metrics", {}).get("peak_process_memory_mb"),
                "vram": run.get("metrics", {}).get("peak_vram_mb"),
                "vram_source": run.get("metrics", {}).get("vram_measurement_source", ""),
                "vram_note": run.get("metrics", {}).get("vram_measurement_note", ""),
                "errors": len(run.get("errors", [])),
                "transcript": transcript,
                "word_error": score.get("normalized_wer"),
                "strict_word_error": score.get("strict_wer"),
                "accuracy_rank": score.get("balanced_rank"),
            }
        )
    unsupported_asr_count = len(
        [
            model
            for model in data.get("unsupported_models", [])
            if model.get("category") != "reference_llm" and model.get("adapter_name") != "gguf_llm_reference"
        ]
    )
    return {
        "duration_seconds": data.get("source", {}).get("duration_seconds"),
        "chunks": len(data.get("chunk_plan", {}).get("chunks", [])),
        "reference_text": reference_text,
        "reference_source": reference_source,
        "errors": [
            {
                "stage": error.get("stage", "unknown") if isinstance(error, dict) else "unknown",
                "message": error.get("message", "") if isinstance(error, dict) else str(error),
                "status": error.get("status", "failed") if isinstance(error, dict) else "failed",
            }
            for error in data.get("errors", [])
        ],
        "runs": runs,
        "unsupported_count": unsupported_asr_count,
    }


def _fmt(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    if value is None:
        return "n/a"
    return str(value)


def _fmt_seconds(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f} seconds"
    return "n/a"


def _fmt_memory_mb(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "unavailable"
    if value >= 1024:
        return f"{value / 1024:.2f} GB"
    return f"{value:.1f} MB"


def _build_records(payload: dict, report_dir: Path) -> list[dict]:
    records = []
    for item in payload.get("files", []):
        output = item.get("output_path")
        link = ""
        if output:
            compare = Path(output) / "compare_scored.html"
            if not compare.exists():
                compare = Path(output) / "compare.html"
            try:
                href = compare.resolve().relative_to(report_dir.resolve()).as_posix()
            except ValueError:
                href = compare.resolve().as_uri()
            link = f'<a href="{_safe(href)}">Open report</a>'
        status = item.get("status", "unknown")
        summary = _load_result_summary(output)
        records.append(
            {
                "name": Path(str(item.get("source_path", ""))).name,
                "status": status,
                "source_path": str(item.get("source_path", "")),
                "output_path": str(item.get("output_path", "")),
                "report_link": link,
                "summary": summary,
            }
        )
    return records


def _manual_override_prompt(records: list[dict]) -> str:
    lines = [
        "Easy ASR Bench Manual LLM Override Prompt",
        "",
        "Paste this into ChatGPT, Claude, or another LLM if you want a manual corrected reference.",
        "",
        "Output format:",
        "- Return only the corrected transcript text for each file.",
        "- Do not include a title, explanation, confidence score, bullet list, JSON, Markdown, or notes section.",
        "- If the audio clearly has multiple speakers, use short speaker labels only when the ASR evidence supports them, such as Speaker 1: and Speaker 2:.",
        "- If speakers are not clear, do not invent speaker labels.",
        "- Use normal paragraph breaks for readability when the transcript is long; otherwise keep it as one clean transcript.",
        "- Always choose the best-supported wording from the ASR outputs; do not use uncertainty placeholders.",
        "",
        "Correction rules:",
        "- Infer the most likely spoken words from the agreement and disagreements between ASR models.",
        "- Preserve the speaker's intent, wording style, and register.",
        "- The source may be a podcast, interview, lecture, news clip, comedy bit, meeting, dialogue scene, phone call, informal chat, scripted speech, technical discussion, or explicit/NSFW speech.",
        "- Keep topic-specific terms, dialogue flow, names, acronyms, numbers, quoted phrases, filler words, dialect, slang, jokes, informal grammar, incomplete sentences, false starts, cut-off thoughts, and profanity when the ASR evidence supports them.",
        "- Do not sanitize, formalize, summarize, paraphrase, complete fragments into polished grammar, or make the text sound more polished than the speaker.",
        "- Use punctuation and capitalization to improve readability, but do not let punctuation edits change the spoken meaning.",
        "- Prefer words supported by multiple ASR outputs. If one model is clearly garbled and another is coherent, use the coherent wording.",
        "- If all models disagree, still choose the most plausible wording from the available ASR evidence.",
        "- Do not invent missing speech, facts, speaker labels, or context that is not supported by the ASR outputs.",
        "After you get the corrected text, paste the matching file's text into the Corrected reference box in final_results.html.",
        "",
    ]
    for index, record in enumerate(records, 1):
        summary = record.get("summary", {})
        lines.extend(
            [
                f"FILE {index}: {record.get('name', 'unnamed file')}",
                f"Duration: {_fmt_seconds(summary.get('duration_seconds'))}",
                "",
                "ASR MODEL TRANSCRIPTS:",
            ]
        )
        for run in summary.get("runs", []):
            lines.extend(
                [
                    "",
                    f"Model: {run.get('display_name', 'Unnamed model')} ({run.get('backend', 'unknown')} / {run.get('precision', 'unknown')})",
                    str(run.get("transcript", "")).strip() or "[No transcript produced]",
                ]
            )
        lines.extend(["", "Return corrected plain text for this file under this heading:", f"CORRECTED FILE {index}: {record.get('name', 'unnamed file')}", "", "-" * 72, ""])
    return "\n".join(lines).rstrip() + "\n"


def _discover_batch_runs(output_root: Path) -> list[dict]:
    runs = []
    if not output_root.exists():
        return runs
    for folder in sorted(output_root.glob("batch__*"), key=lambda path: path.name, reverse=True):
        if not folder.is_dir():
            continue
        batch_path = folder / "_data" / "batch.json"
        index_path = folder / "final_results.html"
        if not batch_path.exists() or not index_path.exists():
            continue
        try:
            payload = json.loads(batch_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        files = payload.get("files", [])
        completed = sum(1 for item in files if item.get("status") == "done")
        runs.append(
            {
                "id": folder.name,
                "created_local": payload.get("created_local", ""),
                "file_count": len(files),
                "completed": completed,
                "failed": len(files) - completed,
                "href": f"{folder.name}/{index_path.name}",
            }
        )
    return runs


def _write_results_index(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    runs = _discover_batch_runs(output_root)
    write_json_atomic(output_root / "report-runs.json", {"schema": "easy_asr_bench.report_runs.v1", "runs": runs})
    (output_root / "index.html").write_text(render_results_index_html(runs), encoding="utf-8", newline="\n")


def render_results_index_html(runs: list[dict]) -> str:
    embedded = json_for_script_tag({"schema": "easy_asr_bench.report_runs.v1", "runs": runs})
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Easy ASR Bench Results</title>
<style>
:root {{ --bg:#f6f7f9; --panel:#fff; --ink:#182334; --muted:#627083; --line:#d6deea; --blue:#1f5fae; --blueSoft:#eaf2ff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Segoe UI, Arial, sans-serif; color:var(--ink); background:var(--bg); font-size:14px; }}
header {{ padding:22px 28px; background:#fff; border-bottom:1px solid var(--line); }}
h1 {{ margin:0; font-size:24px; line-height:1.2; }}
p {{ margin:5px 0 0; color:var(--muted); }}
main {{ max-width:980px; margin:0 auto; padding:24px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; }}
.control {{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:10px; align-items:end; margin-bottom:16px; }}
label {{ display:block; color:var(--muted); margin-bottom:6px; }}
select {{ width:100%; min-height:40px; border:1px solid var(--line); border-radius:6px; padding:8px 10px; font:inherit; background:#fff; }}
button {{ min-height:40px; border:0; border-radius:6px; padding:8px 14px; background:var(--blue); color:#fff; font:inherit; cursor:pointer; }}
.summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }}
.summary div {{ border:1px solid var(--line); border-radius:6px; background:#fbfcfe; padding:12px; }}
.summary span {{ display:block; color:var(--muted); font-size:12px; text-transform:uppercase; }}
.summary strong {{ display:block; margin-top:5px; font-size:20px; }}
.empty {{ color:var(--muted); }}
@media (max-width:640px) {{ .control {{ grid-template-columns:1fr; }} main {{ padding:14px; }} header {{ padding:18px; }} }}
</style>
</head>
<body>
<header><h1>Easy ASR Bench Results</h1><p>Select a benchmark run by date and open its report.</p></header>
<main>
  <section class="panel">
    <div class="control">
      <div><label for="runSelect">Benchmark run</label><select id="runSelect"></select></div>
      <button onclick="openSelectedRun()">Open report</button>
    </div>
    <div id="runSummary" class="summary"></div>
  </section>
</main>
<script type="application/json" id="runs-json">{embedded}</script>
<script>
const payload = JSON.parse(document.getElementById('runs-json').textContent);
const runs = payload.runs || [];
function labelForRun(run) {{
  const date = run.created_local ? new Date(run.created_local) : null;
  const pretty = date && !Number.isNaN(date.getTime()) ? date.toLocaleString() : run.id;
  return `${{pretty}} - ${{run.file_count}} file${{run.file_count === 1 ? '' : 's'}}`;
}}
function renderRuns() {{
  const select = document.getElementById('runSelect');
  if (!runs.length) {{
    select.innerHTML = '<option>No benchmark runs found</option>';
    document.getElementById('runSummary').innerHTML = '<p class="empty">Run a batch benchmark to create a report.</p>';
    return;
  }}
  select.innerHTML = runs.map((run, index) => `<option value="${{index}}">${{labelForRun(run)}}</option>`).join('');
  select.onchange = renderSummary;
  renderSummary();
}}
function renderSummary() {{
  const run = runs[Number(document.getElementById('runSelect').value || 0)];
  document.getElementById('runSummary').innerHTML = run ? `
    <div><span>Files</span><strong>${{run.file_count}}</strong></div>
    <div><span>Completed</span><strong>${{run.completed}}</strong></div>
    <div><span>Failed</span><strong>${{run.failed}}</strong></div>
  ` : '';
}}
function openSelectedRun() {{
  const run = runs[Number(document.getElementById('runSelect').value || 0)];
  if (run) window.location.href = run.href;
}}
renderRuns();
</script>
</body>
</html>
"""


def render_batch_html(payload: dict, report_dir: Path) -> str:
    records = _build_records(payload, report_dir)
    embedded_records = json_for_script_tag(records)
    embedded = json_for_script_tag(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Easy ASR Bench Batch Report</title>
<style>
:root {{ --bg:#f7f8fb; --panel:#fff; --ink:#142033; --muted:#5e6b7d; --line:#d5dde7; --blue:#1f5fae; --blueSoft:#e8f1ff; --green:#dff3e5; --greenText:#1d6b3a; --red:#ffd9dc; --redText:#9f1d2b; --amber:#fff0bd; --amberText:#765414; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Segoe UI, Arial, sans-serif; background:var(--bg); color:var(--ink); font-size:13px; }}
header {{ padding:16px 24px; background:#fff; color:var(--ink); border-bottom:1px solid var(--line); }}
header {{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }}
header h1 {{ margin:0; font-size:20px; line-height:1.2; }}
header p {{ margin:3px 0 0; color:var(--muted); }}
main {{ max-width:1500px; margin:0 auto; padding:16px 20px 36px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:8px; margin-bottom:12px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:10px; }}
.viewer {{ display:grid; grid-template-columns:minmax(210px,300px) minmax(0,1fr); gap:12px; align-items:start; }}
.overall {{ margin:0 0 14px; }}
.overall h2 {{ margin:0 0 8px; font-size:18px; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:6px; background:#fff; }}
table {{ width:100%; border-collapse:collapse; min-width:780px; }}
th, td {{ border-bottom:1px solid var(--line); padding:9px 10px; text-align:left; vertical-align:top; }}
th {{ background:#eef3f9; font-size:12px; text-transform:uppercase; color:#536176; }}
td strong {{ display:block; }}
.file-panel {{ display:flex; flex-direction:column; gap:8px; }}
.file-panel input {{ border:1px solid var(--line); border-radius:6px; padding:8px 10px; min-height:36px; width:100%; }}
.file-list {{ display:flex; flex-direction:column; gap:8px; }}
.file-button {{ width:100%; text-align:left; background:#fff; color:var(--ink); border:1px solid var(--line); border-radius:6px; padding:10px; cursor:pointer; }}
.file-button.active {{ border-color:var(--blue); background:var(--blueSoft); }}
.file-button strong {{ display:block; overflow-wrap:anywhere; font-size:13px; margin:5px 0 0; }}
.file-card {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:14px; min-width:0; }}
.file-card h2 {{ margin:8px 0 6px; font-size:18px; line-height:1.25; overflow-wrap:anywhere; }}
.file-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:10px; }}
.file-card h3 {{ margin:12px 0 6px; font-size:14px; }}
.path {{ color:var(--muted); font-size:12px; overflow-wrap:anywhere; margin:0 0 10px; }}
details {{ margin:0 0 8px; }}
summary {{ color:var(--muted); cursor:pointer; width:max-content; }}
.facts {{ display:flex; flex-wrap:wrap; gap:8px; margin:0 0 10px; }}
.facts span {{ background:#eef2f7; border-radius:999px; padding:3px 8px; font-size:12px; color:#27384f; }}
.reference-panel {{ border:1px solid #ead89e; background:#fffaf0; border-radius:6px; padding:10px; margin:0 0 12px; }}
.reference-editor {{ width:100%; min-height:92px; border:1px solid var(--line); border-radius:6px; padding:8px; font-family:Consolas, monospace; resize:vertical; }}
.reference-actions {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:8px; }}
.reference-status {{ color:#0f6b3f; background:#dff6e8; border-radius:999px; padding:3px 8px; font-size:12px; font-weight:600; }}
.reference-status.updated-flash {{ animation:referenceFlash 850ms ease-out; }}
@keyframes referenceFlash {{
  0% {{ background:#1fba6a; color:#fff; box-shadow:0 0 0 0 rgba(31,186,106,.55); }}
  45% {{ background:#1fba6a; color:#fff; box-shadow:0 0 0 5px rgba(31,186,106,.18); }}
  100% {{ background:#dff6e8; color:#0f6b3f; box-shadow:0 0 0 0 rgba(31,186,106,0); }}
}}
.small-button {{ background:var(--blue); color:#fff; border:0; border-radius:6px; padding:7px 10px; margin-top:8px; cursor:pointer; }}
.secondary-button {{ background:#eef2f7; color:#27384f; border:1px solid var(--line); border-radius:6px; padding:7px 10px; margin-top:8px; cursor:pointer; font:inherit; }}
.error-summary {{ border:1px solid #f0b4ba; background:#fff4f5; border-radius:6px; padding:8px 10px; margin:0 0 10px; color:#7f1d2b; }}
.error-summary strong {{ display:block; font-size:12px; text-transform:uppercase; margin-bottom:3px; }}
.error-summary span {{ display:block; overflow-wrap:anywhere; }}
.toolbar {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:0 0 14px; }}
.toolbar input {{ border:1px solid var(--line); border-radius:6px; padding:8px 10px; min-height:36px; min-width:260px; }}
.model-section-head {{ display:flex; align-items:center; justify-content:space-between; gap:10px; margin:14px 0 8px; }}
.model-section-head h3 {{ margin:0; }}
.model-pager {{ display:flex; align-items:center; gap:8px; }}
.compare-controls {{ display:grid; grid-template-columns:minmax(220px,1fr) minmax(220px,1fr); align-items:end; gap:10px; margin:0 0 12px; }}
.compare-controls label {{ display:flex; flex-direction:column; gap:4px; color:var(--muted); font-size:12px; min-width:0; }}
.compare-controls select {{ min-height:36px; width:100%; min-width:0; border:1px solid var(--line); border-radius:6px; padding:5px 8px; background:#fff; font:inherit; }}
.model-count {{ color:var(--muted); }}
.icon-button {{ width:34px; height:34px; border:1px solid var(--line); border-radius:50%; background:#fff; color:var(--blue); font-size:20px; line-height:1; cursor:pointer; display:inline-flex; align-items:center; justify-content:center; }}
.icon-button:disabled {{ color:#aab4c1; cursor:default; background:#f3f5f8; }}
.text-button {{ min-height:34px; border:1px solid var(--line); border-radius:6px; padding:5px 10px; background:#fff; color:var(--blue); cursor:pointer; font:inherit; }}
.label {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
.value {{ font-size:18px; margin-top:4px; }}
a {{ color:var(--blue); font-weight:600; }}
.muted {{ color:var(--muted); font-size:12px; }}
.status {{ display:inline-block; border-radius:999px; padding:2px 8px; font-size:12px; font-weight:600; }}
.status.ok {{ background:var(--green); color:var(--greenText); }}
.status.fail {{ background:var(--red); color:var(--redText); }}
.run-status {{ color:var(--muted); font-size:12px; white-space:nowrap; }}
.run-status.fail {{ color:var(--redText); font-weight:600; }}
pre {{ white-space:pre-wrap; word-break:break-word; background:#f7f8fa; border:1px solid var(--line); border-radius:6px; padding:8px; }}
.reference {{ margin:0 0 8px; }}
.model-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:10px; }}
.model-grid.compare-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
.model-card {{ border:1px solid var(--line); border-radius:6px; padding:10px; background:#fff; min-width:0; }}
.model-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:10px; margin:0 0 8px; }}
.model-head h4 {{ margin:0; font-size:14px; overflow-wrap:anywhere; }}
.model-head span {{ color:var(--muted); font-size:12px; }}
.transcript {{ margin:0; max-height:220px; overflow:auto; min-height:92px; white-space:pre-wrap; word-break:break-word; background:#f7f8fa; border:1px solid var(--line); border-radius:6px; padding:8px; font-family:Consolas, monospace; line-height:1.65; }}
.word-ok {{ color:var(--ink); }}
.word-replace, .word-insert, .word-delete {{ background:var(--red); color:var(--redText); border-radius:3px; padding:1px 3px; }}
.word-delete {{ font-family:Segoe UI, Arial, sans-serif; font-size:12px; }}
.word-format {{ background:var(--amber); color:var(--amberText); border-radius:3px; padding:1px 3px; }}
.word-insert {{ text-decoration:line-through; }}
.word-note {{ font-family:Segoe UI, Arial, sans-serif; font-size:11px; margin-left:2px; }}
.metric-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:8px; margin-top:8px; }}
.metric-row span {{ border:1px solid var(--line); border-radius:6px; padding:7px; background:#fbfcfe; min-width:0; overflow-wrap:anywhere; }}
.metric-row b {{ display:block; font-size:13px; min-width:0; overflow-wrap:anywhere; }}
.metric-row small {{ display:block; color:var(--muted); min-width:0; overflow-wrap:anywhere; }}
.metric-alert {{ display:inline-block; margin-top:4px; max-width:100%; border-radius:999px; padding:2px 6px; background:#fff2c7; color:#794b00; font-size:11px; font-weight:700; font-style:normal; line-height:1.25; white-space:normal; overflow-wrap:anywhere; }}
.advanced {{ margin-top:12px; }}
@media (max-width:900px) {{ .viewer {{ grid-template-columns:1fr; }} }}
@media (max-width:900px) {{ .compare-controls {{ grid-template-columns:1fr; }} .model-section-head {{ align-items:flex-start; flex-direction:column; }} .model-grid.compare-grid {{ grid-template-columns:1fr; }} }}
@media (max-width:760px) {{ header {{ padding:16px; }} main {{ padding:12px 12px 32px; }} }}
</style>
</head>
<body>
<header>
  <div><h1>Easy ASR Bench</h1>
  <p>Find the best local transcription model for these files.</p></div>
  <a href="https://github.com/rollingedit/Easy-ASR-Bench">GitHub</a>
</header>
<main>
  <div class="grid">
    <div class="card"><div class="label">Files</div><div class="value">{len(payload.get("files", []))}</div></div>
    <div class="card"><div class="label">Completed</div><div class="value">{sum(1 for item in payload.get("files", []) if item.get("status") == "done")}</div></div>
    <div class="card"><div class="label">Failed</div><div class="value">{sum(1 for item in payload.get("files", []) if item.get("status") != "done")}</div></div>
  </div>
  <section id="overallRanking" class="card overall"></section>
  <div class="viewer">
    <aside class="file-panel"><input id="filter" type="search" placeholder="Find a file or model" oninput="selectedIndex=0;renderBatch()"><div id="fileList" class="file-list"></div></aside>
    <section id="selectedFile"></section>
  </div>
  <details class="advanced"><summary>Raw batch data</summary><pre id="raw"></pre></details>
</main>
<script type="application/json" id="batch-json">{embedded}</script>
<script type="application/json" id="batch-records">{embedded_records}</script>
<script>
const payload = JSON.parse(document.getElementById('batch-json').textContent);
const records = JSON.parse(document.getElementById('batch-records').textContent);
let selectedIndex = 0;
const customReferences = {{}};
const customReferenceSources = {{}};
const editedReferences = {{}};
let currentReferenceKey = '';
let selectedModelKey = 'all';
let compareLeftKey = '';
let compareRightKey = '';
let modelPage = 0;
let referenceUpdateTimer = 0;
const modelsPerPage = 3;
function escapeHtml(text) {{
  return String(text ?? '').replace(/[&<>]/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[ch]));
}}
function fmtNumber(n, digits=2) {{
  const value = Number(n);
  return Number.isFinite(value) ? value.toFixed(digits) : 'not available';
}}
function fmtSeconds(n) {{
  const value = Number(n);
  return Number.isFinite(value) ? `${{value.toFixed(2)}} seconds` : 'not available';
}}
function fmtMemoryMb(n) {{
  const value = Number(n);
  if (!Number.isFinite(value)) return 'not available';
  if (value >= 1024) return `${{(value / 1024).toFixed(2)}} GB`;
  return `${{value.toFixed(1)}} MB`;
}}
function fmtPercent(n) {{
  const value = Number(n);
  return Number.isFinite(value) ? `${{(value * 100).toFixed(1)}}%` : 'not scored';
}}
function wordErrorFlag(n) {{
  const value = Number(n);
  return Number.isFinite(value) && value > 1 ? '<em class="metric-alert">Very high</em>' : '';
}}
function statusClass(status) {{
  return status === 'done' ? 'ok' : 'fail';
}}
function referenceWordCount(text) {{
  return wordTokens(text).length;
}}
function modelKey(run) {{
  return [run.candidate_id || run.display_name || '', run.backend || '', run.precision || ''].join('|');
}}
function strictTokens(text) {{
  return String(text || '').trim().split(/\\s+/).filter(Boolean);
}}
function editDistanceTokens(left, right) {{
  const dp = Array(left.length + 1).fill(null).map(() => Array(right.length + 1).fill(0));
  for (let i = 1; i <= left.length; i++) dp[i][0] = i;
  for (let j = 1; j <= right.length; j++) dp[0][j] = j;
  for (let i = 1; i <= left.length; i++) for (let j = 1; j <= right.length; j++) {{
    const cost = left[i - 1] === right[j - 1] ? 0 : 1;
    dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
  }}
  return dp[left.length][right.length];
}}
function werFromTokens(referenceTokens, hypothesisTokens) {{
  if (!referenceTokens.length) return null;
  return editDistanceTokens(referenceTokens, hypothesisTokens) / referenceTokens.length;
}}
function scoreTranscript(referenceText, transcriptText) {{
  if (!String(referenceText || '').trim()) return {{ word_error: null, strict_word_error: null }};
  return {{
    word_error: werFromTokens(wordTokens(referenceText).map(token => token.norm), wordTokens(transcriptText).map(token => token.norm)),
    strict_word_error: werFromTokens(strictTokens(referenceText), strictTokens(transcriptText)),
  }};
}}
function referenceKeyForRecord(record) {{
  return record.output_path || record.source_path || record.name;
}}
function effectiveReferenceForRecord(record) {{
  return customReferences[referenceKeyForRecord(record)] ?? record.summary?.reference_text ?? '';
}}
function effectiveRunsForRecord(record) {{
  const referenceText = effectiveReferenceForRecord(record);
  const runs = record.summary?.runs || [];
  const rescored = runs.map(run => ({{ ...run, ...scoreTranscript(referenceText, run.transcript || '') }}));
  rescored.sort((a, b) => Number(a.word_error ?? Infinity) - Number(b.word_error ?? Infinity));
  rescored.forEach((run, index) => {{ run.accuracy_rank = Number.isFinite(Number(run.word_error)) ? index + 1 : null; }});
  return runs.map(run => rescored.find(item => modelKey(item) === modelKey(run)) || run);
}}
function aggregateModelRanking(sourceRecords) {{
  const models = new Map();
  for (const record of sourceRecords) {{
    const referenceText = effectiveReferenceForRecord(record);
    const referenceWords = Math.max(1, referenceWordCount(referenceText));
    const scoredRuns = effectiveRunsForRecord(record).filter(run => Number.isFinite(Number(run.word_error)));
    if (!scoredRuns.length) continue;
    const bestWer = Math.min(...scoredRuns.map(run => Number(run.word_error)));
    for (const run of scoredRuns) {{
      const key = modelKey(run);
      if (!models.has(key)) {{
        models.set(key, {{
          display_name: run.display_name || 'Unnamed model',
          backend: run.backend || 'unknown',
          precision: run.precision || 'unknown',
          files: 0,
          wins: 0,
          weightedWer: 0,
          weightedStrictDelta: 0,
          weight: 0,
          totalSeconds: 0,
          ramPeak: 0,
          vramPeak: 0,
          runtimeIssues: 0,
        }});
      }}
      const item = models.get(key);
      const wer = Number(run.word_error);
      const strictWer = Number(run.strict_word_error);
      item.files += 1;
      item.weightedWer += wer * referenceWords;
      if (Number.isFinite(strictWer)) item.weightedStrictDelta += Math.max(0, strictWer - wer) * referenceWords;
      item.weight += referenceWords;
      item.totalSeconds += Number(run.transcription_seconds || 0);
      item.ramPeak = Math.max(item.ramPeak, Number(run.ram || 0));
      item.vramPeak = Math.max(item.vramPeak, Number(run.vram || 0));
      item.runtimeIssues += Number(run.errors || 0);
      if (wer === bestWer) item.wins += 1;
    }}
  }}
  return [...models.values()].map(item => ({{
    ...item,
    weightedWerAverage: item.weight ? item.weightedWer / item.weight : null,
    weightedStrictDeltaAverage: item.weight ? item.weightedStrictDelta / item.weight : null,
  }})).sort((a, b) => {{
    const accuracy = Number(a.weightedWerAverage ?? Infinity) - Number(b.weightedWerAverage ?? Infinity);
    if (accuracy) return accuracy;
    const wins = b.wins - a.wins;
    if (wins) return wins;
    return a.totalSeconds - b.totalSeconds;
  }});
}}
function renderOverallRanking(sourceRecords) {{
  const ranking = aggregateModelRanking(sourceRecords);
  if (!ranking.length) {{
    document.getElementById('overallRanking').innerHTML = '<h2>Overall model ranking</h2><p class="muted">No scored corrected-reference results are available yet.</p>';
    return;
  }}
  const rows = ranking.map((model, index) => `<tr>
    <td><strong>#${{index + 1}} ${{escapeHtml(model.display_name)}}</strong><span class="muted">${{escapeHtml(model.backend)}} / ${{escapeHtml(model.precision)}}</span></td>
    <td>${{fmtPercent(model.weightedWerAverage)}}<br><span class="muted">weighted by reference words</span></td>
    <td>${{fmtPercent(model.weightedStrictDeltaAverage)}}<br><span class="muted">not scored</span></td>
    <td>${{model.wins}} / ${{model.files}}</td>
    <td>${{fmtSeconds(model.totalSeconds)}}</td>
    <td>${{fmtMemoryMb(model.ramPeak)}}</td>
    <td>${{fmtMemoryMb(model.vramPeak)}}</td>
    <td>${{model.runtimeIssues}}</td>
  </tr>`).join('');
  document.getElementById('overallRanking').innerHTML = `<h2>Overall model ranking</h2><div class="table-wrap"><table><thead><tr><th>Model</th><th>Word error rate</th><th>Punctuation/capitalization</th><th>File wins</th><th>Total transcription time</th><th>RAM peak</th><th>VRAM / GPU memory peak</th><th>Runtime issues</th></tr></thead><tbody>${{rows}}</tbody></table></div><p class="muted">Score uses word error rate weighted by corrected-reference word count. Yellow punctuation/capitalization differences are shown for review only and do not affect the score.</p>`;
}}
function normalizeWord(word) {{
  return String(word || '').normalize('NFKC').toLocaleLowerCase().replace(/[^\\p{{L}}\\p{{N}}\\p{{M}}']+/gu, '');
}}
function formatInsensitiveWord(word) {{
  return String(word || '').normalize('NFKC').toLocaleLowerCase().replace(/[^\\p{{L}}\\p{{N}}\\p{{M}}]+/gu, '');
}}
function wordTokens(text) {{
  return String(text || '').trim().split(/\\s+/).filter(Boolean).map(raw => ({{ raw, norm: normalizeWord(raw) }})).filter(token => token.norm);
}}
function wordErrorAgainstReference(referenceText, transcriptText) {{
  const alignment = alignWords(referenceText, transcriptText);
  const refCount = alignment.filter(item => item.op !== 'insert').length;
  const edits = alignment.filter(item => item.op !== 'equal' && item.op !== 'format').length;
  return refCount ? edits / refCount : null;
}}
function strictDeltaText(run) {{
  const normalized = Number(run.word_error);
  const strict = Number(run.strict_word_error);
  if (!Number.isFinite(normalized) || !Number.isFinite(strict)) return '';
  const delta = Math.max(0, strict - normalized);
  return `${{fmtPercent(delta)}} punctuation/case`;
}}
function alignWords(referenceText, transcriptText) {{
  const ref = wordTokens(referenceText);
  const hyp = wordTokens(transcriptText);
  const dp = Array(ref.length + 1).fill(null).map(() => Array(hyp.length + 1).fill(0));
  const op = Array(ref.length + 1).fill(null).map(() => Array(hyp.length + 1).fill(''));
  for (let i = 1; i <= ref.length; i++) {{ dp[i][0] = i; op[i][0] = 'delete'; }}
  for (let j = 1; j <= hyp.length; j++) {{ dp[0][j] = j; op[0][j] = 'insert'; }}
  for (let i = 1; i <= ref.length; i++) for (let j = 1; j <= hyp.length; j++) {{
    const choices = ref[i - 1].norm === hyp[j - 1].norm ? [[dp[i - 1][j - 1], 'equal']] : [[dp[i - 1][j - 1] + 1, 'replace']];
    choices.push([dp[i - 1][j] + 1, 'delete'], [dp[i][j - 1] + 1, 'insert']);
    choices.sort((a, b) => a[0] - b[0]);
    dp[i][j] = choices[0][0];
    op[i][j] = choices[0][1];
  }}
  let i = ref.length, j = hyp.length, out = [];
  while (i > 0 || j > 0) {{
    const action = op[i][j];
    if (action === 'equal') {{ out.push({{ op: 'equal', ref: ref[i - 1]?.raw || '', hyp: hyp[j - 1]?.raw || '' }}); i--; j--; }}
    else if (action === 'replace') {{ out.push({{ op: 'replace', ref: ref[i - 1]?.raw || '', hyp: hyp[j - 1]?.raw || '' }}); i--; j--; }}
    else if (action === 'delete') {{ out.push({{ op: 'delete', ref: ref[i - 1]?.raw || '', hyp: '' }}); i--; }}
    else {{ out.push({{ op: 'insert', ref: '', hyp: hyp[j - 1]?.raw || '' }}); j--; }}
  }}
  return out.reverse();
}}
function renderHighlightedTranscript(referenceText, transcriptText) {{
  if (!String(referenceText || '').trim()) return escapeHtml(transcriptText || 'No transcript was produced.');
  return alignWords(referenceText, transcriptText).map(item => {{
    if (item.op === 'equal') return `<span class="word-ok">${{escapeHtml(item.hyp)}}</span>`;
    if (item.op === 'insert') return `<span class="word-insert">${{escapeHtml(item.hyp)}}<span class="word-note"> extra</span></span>`;
    if (item.op === 'delete') return `<span class="word-delete">missing: ${{escapeHtml(item.ref)}}</span>`;
    if (formatInsensitiveWord(item.ref) === formatInsensitiveWord(item.hyp)) return `<span class="word-format">${{escapeHtml(item.hyp)}}<span class="word-note"> -> ${{escapeHtml(item.ref)}}</span></span>`;
    return `<span class="word-replace">${{escapeHtml(item.hyp)}}<span class="word-note"> -> ${{escapeHtml(item.ref)}}</span></span>`;
  }}).join(' ');
}}
function filteredRecords() {{
  const query = document.getElementById('filter').value.trim().toLowerCase();
  if (!query) return records;
  return records.filter(record => JSON.stringify(record).toLowerCase().includes(query));
}}
function modelCard(run, referenceText) {{
  const rank = run.accuracy_rank ? `#${{run.accuracy_rank}}` : 'not scored';
  const wordError = fmtPercent(run.word_error);
  const cleanup = strictDeltaText(run);
  const issueCount = Number(run.errors || 0);
  const issueLabel = issueCount === 0 ? 'Run completed' : issueCount === 1 ? '1 runtime problem' : `${{issueCount}} runtime problems`;
  return `<article class="model-card">
    <div class="model-head"><div><h4>${{escapeHtml(run.display_name || 'Unnamed model')}}</h4><span>${{escapeHtml(run.backend || 'unknown')}} / ${{escapeHtml(run.precision || 'unknown')}}</span></div><span class="run-status ${{issueCount ? 'fail' : ''}}">${{escapeHtml(issueLabel)}}</span></div>
    <div class="transcript">${{renderHighlightedTranscript(referenceText, run.transcript || 'No transcript was produced.')}}</div>
    <div class="metric-row">
      <span><b>${{escapeHtml(rank)}}</b><small>Accuracy rank</small></span>
      <span><b>${{escapeHtml(wordError)}}</b>${{wordErrorFlag(run.word_error)}}<small>Word error rate</small></span>
      <span><b>${{escapeHtml(cleanup || 'not scored')}}</b><small>Punctuation/case</small></span>
      <span><b>${{fmtSeconds(run.transcription_seconds)}}</b><small>Transcription time</small></span>
      <span title="Peak system RAM for this model run; VRAM is listed separately."><b>${{fmtMemoryMb(run.ram)}}</b><small>RAM peak</small></span>
      <span title="${{escapeHtml(run.vram_note || 'VRAM/GPU memory is reported separately. On integrated GPUs it may be shared system memory, so do not add it to RAM as a total.')}}"><b>${{fmtMemoryMb(run.vram)}}</b><small>VRAM / GPU memory peak</small></span>
    </div>
  </article>`;
}}
function perFileTopModels(runs) {{
  return (runs || [])
    .filter(run => Number.isFinite(Number(run.word_error)))
    .sort((a, b) => {{
      const accuracy = Number(a.word_error) - Number(b.word_error);
      if (accuracy) return accuracy;
      return Number(a.transcription_seconds || Infinity) - Number(b.transcription_seconds || Infinity);
    }})
    .slice(0, 3);
}}
function renderPerFileTopModels(runs) {{
  const top = perFileTopModels(runs);
  if (!top.length) return '<p class="muted">No per-file accuracy ranking is available for this file yet.</p>';
  return `<div class="metric-row">${{top.map((run, index) => `<span><b>#${{index + 1}} ${{escapeHtml(run.display_name || 'Unnamed model')}}</b><small>${{fmtPercent(run.word_error)}} word error rate, ${{strictDeltaText(run) || '0.0% punctuation/case'}}, ${{fmtSeconds(run.transcription_seconds)}}</small></span>`).join('')}}</div>`;
}}
function renderErrors(errors) {{
  return (errors || []).map(error => `<div class="error-summary"><strong>${{escapeHtml(error.stage || 'unknown')}}</strong><span>${{escapeHtml(error.message || '')}}</span></div>`).join('');
}}
function firstOtherModelKey(runs, selectedKey) {{
  const other = runs.find(run => modelKey(run) !== selectedKey);
  return other ? modelKey(other) : '';
}}
function normalizeCompareKeys(runs) {{
  if (runs.length < 2) {{
    selectedModelKey = 'all';
    compareLeftKey = runs[0] ? modelKey(runs[0]) : '';
    compareRightKey = '';
    return;
  }}
  if (!compareLeftKey || !runs.some(run => modelKey(run) === compareLeftKey)) compareLeftKey = modelKey(runs[0]);
  if (!compareRightKey || !runs.some(run => modelKey(run) === compareRightKey) || compareRightKey === compareLeftKey) {{
    compareRightKey = firstOtherModelKey(runs, compareLeftKey);
  }}
  if (compareLeftKey === compareRightKey) selectedModelKey = 'all';
}}
function selectCompareLeft(value) {{
  compareLeftKey = value;
  compareRightKey = firstOtherModelKey(effectiveRunsForRecord(filteredRecords()[selectedIndex] || {{}}), compareLeftKey);
  selectedModelKey = compareRightKey ? 'compare' : 'all';
  modelPage = 0;
  renderBatch();
}}
function selectCompareRight(value) {{
  compareRightKey = value;
  if (compareLeftKey === compareRightKey) compareLeftKey = firstOtherModelKey(effectiveRunsForRecord(filteredRecords()[selectedIndex] || {{}}), compareRightKey);
  selectedModelKey = compareLeftKey && compareLeftKey !== compareRightKey ? 'compare' : 'all';
  modelPage = 0;
  renderBatch();
}}
function buildManualPromptForRecord(record) {{
  const summary = record.summary || {{}};
  const lines = [
    'Easy ASR Bench manual corrected-reference prompt',
    '',
    'Task: Create one corrected transcript from the ASR outputs below.',
    '',
    'Output format:',
    '- Return only the corrected transcript text.',
    '- Do not include a title, explanation, confidence score, bullet list, JSON, Markdown, or notes section.',
    '- If the audio clearly has multiple speakers, use short speaker labels only when the ASR evidence supports them, such as Speaker 1: and Speaker 2:.',
    '- If speakers are not clear, do not invent speaker labels.',
    '- Use normal paragraph breaks for readability when the transcript is long; otherwise keep it as one clean transcript.',
    '- Always choose the best-supported wording from the ASR outputs; do not use uncertainty placeholders.',
    '',
    'Correction rules:',
    '- Preserve speaker intent, dialogue flow, tone, speaker wording, topic-specific terms, names, acronyms, numbers, quoted phrases, slang, filler words, jokes, informal grammar, incomplete sentences, false starts, cut-off thoughts, and profanity when supported by the ASR evidence.',
    '- Preserve topic-specific terms, dialogue flow, names, acronyms, and quoted phrases when the ASR evidence supports them.',
    '- Do not sanitize, formalize, summarize, paraphrase, complete fragments into polished grammar, or invent unsupported speech.',
    '- Use punctuation and capitalization for readability without changing meaning.',
    '- Prefer wording supported by multiple ASR outputs.',
    '- If one ASR output is coherent and another is garbled, use the coherent wording.',
    '',
    `File: ${{record.name || 'unnamed file'}}`,
    `Duration: ${{fmtSeconds(summary.duration_seconds)}}`,
    '',
    'ASR outputs:'
  ];
  for (const run of summary.runs || []) {{
    lines.push('', `Model: ${{run.display_name || 'Unnamed model'}} (${{run.backend || 'unknown'}} / ${{run.precision || 'unknown'}})`, String(run.transcript || '[No transcript produced]').trim());
  }}
  return lines.join('\\n');
}}
async function copyManualPrompt(recordIndex) {{
  const record = filteredRecords()[recordIndex];
  if (!record) return;
  const text = buildManualPromptForRecord(record);
  try {{
    await navigator.clipboard.writeText(text);
    document.getElementById('manualPromptStatus').textContent = 'Prompt copied';
  }} catch (error) {{
    document.getElementById('manualPromptStatus').textContent = 'Copy failed; use browser clipboard permissions and try again';
  }}
}}
function referenceSourceLabel(referenceSource, referenceKey) {{
  if (editedReferences[referenceKey]) return '';
  return referenceSource ? `<span id="referenceSourceLabel" class="muted">Local LLM: ${{escapeHtml(referenceSource.replace(/^Local LLM:\\s*/, ''))}}</span>` : '<span id="referenceSourceLabel" class="muted">Local LLM: none used</span>';
}}
function referenceEditStatus(referenceKey) {{
  return editedReferences[referenceKey] ? 'Results updated from pasted text' : 'Edits update results automatically';
}}
function markReferenceEdited() {{
  const editor = document.getElementById('referenceEditor');
  customReferences[currentReferenceKey] = editor ? editor.value : '';
  editedReferences[currentReferenceKey] = true;
  customReferenceSources[currentReferenceKey] = '';
  const label = document.getElementById('referenceSourceLabel');
  if (label) label.textContent = '';
  const status = document.getElementById('referenceEditStatus');
  if (status) {{
    status.textContent = referenceEditStatus(currentReferenceKey);
    status.classList.remove('updated-flash');
    void status.offsetWidth;
    status.classList.add('updated-flash');
  }}
  clearTimeout(referenceUpdateTimer);
  referenceUpdateTimer = setTimeout(() => renderBatch(), 250);
}}
function renderSelected(record) {{
  if (!record) return '<p class="muted">No files match this search.</p>';
  const summary = record.summary || {{}};
  const runs = effectiveRunsForRecord(record);
  normalizeCompareKeys(runs);
  const compareMode = selectedModelKey === 'compare' && runs.length >= 2 && compareLeftKey && compareRightKey && compareLeftKey !== compareRightKey;
  const filteredRuns = compareMode
    ? runs.filter(run => modelKey(run) === compareLeftKey || modelKey(run) === compareRightKey)
    : selectedModelKey === 'all'
      ? runs
      : runs.filter(run => modelKey(run) === selectedModelKey);
  const pageCount = Math.max(1, Math.ceil(filteredRuns.length / modelsPerPage));
  modelPage = Math.min(modelPage, pageCount - 1);
  const pageStart = modelPage * modelsPerPage;
  const visibleRuns = filteredRuns.slice(pageStart, pageStart + modelsPerPage);
  const pageLabel = filteredRuns.length
    ? `Models ${{pageStart + 1}}-${{Math.min(pageStart + modelsPerPage, filteredRuns.length)}} of ${{filteredRuns.length}}`
    : 'No models';
  const compareOptions = runs.map(run => `<option value="${{escapeHtml(modelKey(run))}}">${{escapeHtml(run.display_name || 'Unnamed model')}}</option>`).join('');
  const compareControls = runs.length >= 2
    ? `<div class="compare-controls"><label><span>Compare model A</span><select aria-label="Compare model A" onchange="selectCompareLeft(this.value)">${{compareOptions.replace(`value="${{escapeHtml(compareLeftKey)}}"`, `value="${{escapeHtml(compareLeftKey)}}" selected`)}}</select></label><label><span>Compare model B</span><select aria-label="Compare model B" onchange="selectCompareRight(this.value)">${{compareOptions.replace(`value="${{escapeHtml(compareRightKey)}}"`, `value="${{escapeHtml(compareRightKey)}}" selected`)}}</select></label></div>`
    : '';
  const gridClass = compareMode ? 'model-grid compare-grid' : 'model-grid';
  const referenceKey = referenceKeyForRecord(record);
  currentReferenceKey = referenceKey;
  const referenceText = effectiveReferenceForRecord(record);
  const referenceSource = customReferenceSources[referenceKey] ?? summary.reference_source ?? '';
  const recordIndex = filteredRecords()[selectedIndex] === record ? selectedIndex : 0;
  const chunks = Number(summary.chunks || 0);
  const skipped = Number(summary.unsupported_count || 0);
  const skippedFact = skipped > 0 ? `<span>${{skipped}} skipped ASR ${{skipped === 1 ? 'model' : 'models'}}</span>` : '';
  const reportLink = record.report_link || '';
  return `<article class="file-card">
    <div class="file-head"><div><span class="status ${{statusClass(record.status)}}">${{escapeHtml(record.status)}}</span><h2>${{escapeHtml(record.name || 'Unnamed file')}}</h2></div></div>
    <div class="facts"><span>${{fmtSeconds(summary.duration_seconds)}}</span><span>${{chunks}} audio ${{chunks === 1 ? 'chunk' : 'chunks'}}</span>${{skippedFact}}</div>
    <section class="reference-panel"><h3>Corrected reference</h3>${{referenceSourceLabel(referenceSource, referenceKey)}}<textarea id="referenceEditor" class="reference-editor" aria-label="Corrected reference" placeholder="Paste corrected transcript text here." oninput="markReferenceEdited()">${{escapeHtml(referenceText)}}</textarea><div class="reference-actions"><button class="secondary-button" onclick="resetCurrentReference()">Reset</button><span id="referenceEditStatus" class="reference-status">${{escapeHtml(referenceEditStatus(referenceKey))}}</span></div></section>
    <section class="card"><h3>Manual LLM correction</h3><p class="muted">Copy this file's prompt, paste it into ChatGPT or Claude, then paste the corrected transcript above.</p><button class="small-button" onclick="copyManualPrompt(${{recordIndex}})">Copy prompt for this file</button><span id="manualPromptStatus" class="muted"></span></section>
    ${{renderErrors(summary.errors)}}
    <h3>Best for this file</h3>
    ${{renderPerFileTopModels(runs)}}
    <div class="model-section-head"><h3>Model transcripts</h3><div class="model-pager"><button class="text-button" onclick="selectedModelKey='all';modelPage=0;renderBatch()">Show all</button><span class="model-count">${{escapeHtml(pageLabel)}}</span><button class="icon-button" aria-label="Previous models" title="Previous models" onclick="modelPage=Math.max(0,modelPage-1);renderBatch()" ${{modelPage <= 0 ? 'disabled' : ''}}>&lsaquo;</button><button class="icon-button" aria-label="Next models" title="Next models" onclick="modelPage=Math.min(${{pageCount - 1}},modelPage+1);renderBatch()" ${{modelPage >= pageCount - 1 ? 'disabled' : ''}}>&rsaquo;</button></div></div>
    ${{compareControls}}
    <div class="${{gridClass}}">${{visibleRuns.length ? visibleRuns.map(run => modelCard(run, referenceText)).join('') : '<p class="muted">No model transcripts were written for this file.</p>'}}</div>
    <details class="advanced"><summary>Advanced details</summary>${{reportLink ? `<p>${{reportLink}}</p>` : ''}}<p class="path">${{escapeHtml(record.source_path || '')}}</p><p class="path">${{escapeHtml(record.output_path || '')}}</p></details>
  </article>`;
}}
function resetCurrentReference() {{
  clearTimeout(referenceUpdateTimer);
  delete customReferences[currentReferenceKey];
  delete customReferenceSources[currentReferenceKey];
  delete editedReferences[currentReferenceKey];
  renderBatch();
}}
function renderBatch() {{
  const filtered = filteredRecords();
  renderOverallRanking(filtered);
  selectedIndex = Math.min(selectedIndex, Math.max(0, filtered.length - 1));
  document.getElementById('fileList').innerHTML = filtered.map((record, index) => `<button class="file-button ${{index === selectedIndex ? 'active' : ''}}" onclick="selectedIndex=${{index}};selectedModelKey='all';compareLeftKey='';compareRightKey='';modelPage=0;renderBatch()"><span class="status ${{statusClass(record.status)}}">${{escapeHtml(record.status)}}</span><strong>${{escapeHtml(record.name || 'Unnamed file')}}</strong><span class="muted">${{fmtSeconds(record.summary?.duration_seconds)}}</span></button>`).join('') || '<p class="muted">No files match this search.</p>';
  document.getElementById('selectedFile').innerHTML = renderSelected(filtered[selectedIndex]);
}}
renderBatch();
document.getElementById('raw').textContent = JSON.stringify(payload, null, 2);
</script>
</body>
</html>
"""
