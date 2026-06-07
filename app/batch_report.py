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
    write_json_atomic(report_dir / "batch.json", payload)
    (report_dir / "index.html").write_text(render_batch_html(payload, report_dir), encoding="utf-8", newline="\n")
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
    return {
        "duration_seconds": data.get("source", {}).get("duration_seconds"),
        "chunks": len(data.get("chunk_plan", {}).get("chunks", [])),
        "runs": [
            {
                "display_name": run.get("model", {}).get("display_name", ""),
                "backend": run.get("model", {}).get("backend", ""),
                "precision": run.get("model", {}).get("precision", ""),
                "speed": run.get("metrics", {}).get("audio_seconds_per_wall_second"),
                "ram": run.get("metrics", {}).get("peak_process_memory_mb"),
                "vram": run.get("metrics", {}).get("peak_vram_mb"),
                "errors": len(run.get("errors", [])),
            }
            for run in data.get("runs", [])
        ],
        "unsupported_count": len(data.get("unsupported_models", [])),
    }


def _fmt(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    if value is None:
        return "n/a"
    return str(value)


def render_batch_html(payload: dict, report_dir: Path) -> str:
    records = []
    for item in payload.get("files", []):
        output = item.get("output_path")
        link = ""
        if output:
            compare = Path(output) / "compare.html"
            try:
                href = compare.resolve().relative_to(report_dir.resolve()).as_posix()
            except ValueError:
                href = compare.resolve().as_uri()
            link = f'<a href="{_safe(href)}">Open report</a>'
        status = item.get("status", "unknown")
        status_class = "ok" if status == "done" else "fail"
        summary = _load_result_summary(output)
        model_rows = []
        for run in summary.get("runs", []):
            error_class = "fail" if run.get("errors") else "ok"
            model_rows.append(
                "<tr>"
                f"<td>{_safe(run.get('display_name'))}<br><span class=\"muted\">{_safe(run.get('backend'))} / {_safe(run.get('precision'))}</span></td>"
                f"<td>{_safe(_fmt(run.get('speed')))}</td>"
                f"<td>{_safe(_fmt(run.get('ram')))}</td>"
                f"<td>{_safe(_fmt(run.get('vram')))}</td>"
                f"<td><span class=\"status {error_class}\">{_safe(run.get('errors'))}</span></td>"
                "</tr>"
            )
        model_table = (
            "<div class=\"mini-table\"><table><thead><tr><th>Model</th><th>xRT</th><th>RAM</th><th>VRAM</th><th>Err</th></tr></thead>"
            f"<tbody>{''.join(model_rows)}</tbody></table></div>"
            if model_rows
            else "<p class=\"muted\">No model rows were written for this file.</p>"
        )
        card = (
            f"<article class=\"file-card\" data-search=\"{_safe((str(item.get('source_path', '')) + ' ' + status).lower())}\">"
            f"<div class=\"file-head\"><span class=\"status {status_class}\">{_safe(status)}</span>{link}</div>"
            f"<h2>{_safe(Path(str(item.get('source_path', ''))).name)}</h2>"
            f"<p class=\"path\">{_safe(item.get('source_path'))}</p>"
            f"<div class=\"facts\"><span>{_safe(_fmt(summary.get('duration_seconds')))}s</span><span>{_safe(summary.get('chunks', 'n/a'))} chunks</span><span>{_safe(summary.get('unsupported_count', 0))} skipped</span></div>"
            f"{model_table}"
            "</article>"
        )
        table_row = (
            "<tr>"
            f"<td><span class=\"status {status_class}\">{_safe(status)}</span></td>"
            f"<td>{_safe(item.get('source_path'))}</td>"
            f"<td>{_safe(item.get('output_path') or '')}</td>"
            f"<td>{link}</td>"
            "</tr>"
        )
        records.append({"card": card, "row": table_row})
    embedded_records = json_for_script_tag(records)
    embedded = json_for_script_tag(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Easy ASR Bench Batch Report</title>
<style>
:root {{ --bg:#f5f6f8; --panel:#fff; --ink:#18212f; --muted:#647184; --line:#d8dee8; --blue:#1f5fae; --green:#dff3e5; --greenText:#1d6b3a; --red:#ffd9dc; --redText:#9f1d2b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Segoe UI, Arial, sans-serif; background:var(--bg); color:var(--ink); }}
header {{ padding:22px 32px; background:#18212f; color:white; border-bottom:4px solid #3ba776; }}
header h1 {{ margin:0 0 6px; font-size:26px; line-height:1.15; }}
header p {{ margin:0; color:#d5dce6; }}
main {{ max-width:1400px; margin:0 auto; padding:20px 32px 48px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:16px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
.file-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:14px; margin:0 0 18px; align-items:start; }}
.file-card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; min-width:0; box-shadow:0 1px 2px rgba(24,33,47,.04); }}
.file-card h2 {{ margin:10px 0 4px; font-size:17px; line-height:1.25; overflow-wrap:anywhere; }}
.file-head {{ display:flex; justify-content:space-between; align-items:center; gap:10px; }}
.path {{ color:var(--muted); font-size:12px; overflow-wrap:anywhere; margin:0 0 10px; }}
.facts {{ display:flex; flex-wrap:wrap; gap:8px; margin:0 0 10px; }}
.facts span {{ background:#eef2f7; border-radius:999px; padding:3px 8px; font-size:12px; color:#27384f; }}
.toolbar {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:0 0 14px; }}
.toolbar input {{ border:1px solid var(--line); border-radius:6px; padding:8px 10px; min-height:36px; min-width:260px; }}
.pager {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:0 0 14px; }}
.pager button {{ background:var(--blue); color:white; border:0; border-radius:6px; padding:8px 12px; min-height:34px; cursor:pointer; }}
.label {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
.value {{ font-size:22px; margin-top:4px; }}
.table-wrap {{ overflow:auto; background:var(--panel); border:1px solid var(--line); border-radius:8px; }}
.mini-table {{ overflow:auto; max-height:260px; border:1px solid var(--line); border-radius:8px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ padding:10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ background:#eef2f7; position:sticky; top:0; }}
a {{ color:var(--blue); font-weight:600; }}
.muted {{ color:var(--muted); font-size:12px; }}
.status {{ display:inline-block; border-radius:999px; padding:2px 8px; font-size:12px; font-weight:600; }}
.status.ok {{ background:var(--green); color:var(--greenText); }}
.status.fail {{ background:var(--red); color:var(--redText); }}
pre {{ white-space:pre-wrap; word-break:break-word; background:#f7f8fa; border:1px solid var(--line); border-radius:6px; padding:10px; }}
@media (max-width:760px) {{ header {{ padding:18px 16px; }} main {{ padding:14px 12px 32px; }} }}
</style>
</head>
<body>
<header>
  <h1>Easy ASR Bench Batch Report</h1>
  <p>One overview for every audio or video file processed in this run.</p>
</header>
<main>
  <div class="grid">
    <div class="card"><div class="label">Files</div><div class="value">{len(payload.get("files", []))}</div></div>
    <div class="card"><div class="label">Completed</div><div class="value">{sum(1 for item in payload.get("files", []) if item.get("status") == "done")}</div></div>
    <div class="card"><div class="label">Failed</div><div class="value">{sum(1 for item in payload.get("files", []) if item.get("status") != "done")}</div></div>
  </div>
  <div class="toolbar">
    <input id="filter" type="search" placeholder="Filter by file path or status" oninput="batchPage=0;renderBatch()">
  </div>
  <div class="pager">
    <button onclick="batchPage=Math.max(0,batchPage-1);renderBatch()">Prev</button>
    <span id="pageStatus"></span>
    <button onclick="batchPage=Math.min(totalBatchPages()-1,batchPage+1);renderBatch()">Next</button>
  </div>
  <section id="fileGrid" class="file-grid"></section>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Status</th><th>Source</th><th>Report Folder</th><th>Action</th></tr></thead>
      <tbody id="tableRows"></tbody>
    </table>
  </div>
  <h2>Raw Batch JSON</h2>
  <pre id="raw"></pre>
</main>
<script type="application/json" id="batch-json">{embedded}</script>
<script type="application/json" id="batch-records">{embedded_records}</script>
<script>
const payload = JSON.parse(document.getElementById('batch-json').textContent);
const records = JSON.parse(document.getElementById('batch-records').textContent);
let batchPage = 0;
const batchPageSize = 6;
function filteredRecords() {{
  const query = document.getElementById('filter').value.trim().toLowerCase();
  if (!query) return records;
  return records.filter(record => record.card.toLowerCase().includes(query));
}}
function totalBatchPages() {{
  return Math.max(1, Math.ceil(filteredRecords().length / batchPageSize));
}}
function renderBatch() {{
  const filtered = filteredRecords();
  const total = totalBatchPages();
  batchPage = Math.min(batchPage, total - 1);
  const page = filtered.slice(batchPage * batchPageSize, (batchPage + 1) * batchPageSize);
  document.getElementById('pageStatus').textContent = `Files ${{filtered.length ? batchPage * batchPageSize + 1 : 0}}-${{Math.min(filtered.length, (batchPage + 1) * batchPageSize)}} of ${{filtered.length}}`;
  document.getElementById('fileGrid').innerHTML = page.map(record => record.card).join('') || '<p class="muted">No files match this filter.</p>';
  document.getElementById('tableRows').innerHTML = page.map(record => record.row).join('');
}}
renderBatch();
document.getElementById('raw').textContent = JSON.stringify(payload, null, 2);
</script>
</body>
</html>
"""
