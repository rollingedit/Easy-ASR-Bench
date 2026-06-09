from __future__ import annotations

from .html_json import json_for_script_tag


def build_html_report(results: dict) -> str:
    if results.get("schema") == "easy_asr_bench.scored_report.v1":
        scored = results
        base_results = dict(scored.get("results") or {})
        if scored.get("status") == "scored":
            base_results["reference_scores"] = scored.get("scores", {})
        if scored.get("reference"):
            base_results["reference"] = scored["reference"]
        for key in ("score_type", "score_note", "status"):
            if key in scored:
                output_key = "score_status" if key == "status" else key
                base_results[output_key] = scored[key]
        results = base_results
    embedded = json_for_script_tag(results)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Easy ASR Bench Report</title>
<style>
:root {{ color-scheme: light; --bg:#f4f6f8; --panel:#fff; --ink:#142033; --muted:#5e6b7d; --line:#d5dde7; --blue:#1f5fae; --blue2:#e7f0ff; --red:#ffd9dc; --red2:#9f1d2b; --yellow:#fff1b8; --green:#dff3e5; --green2:#1d6b3a; --violet:#efe9ff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Segoe UI, Arial, sans-serif; background:var(--bg); color:var(--ink); font-size:13px; }}
header {{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; padding:12px 20px; background:#fff; color:var(--ink); border-bottom:1px solid var(--line); }}
header h1 {{ margin:0; font-size:20px; line-height:1.2; letter-spacing:0; }}
header p {{ margin:2px 0 0; color:var(--muted); max-width:900px; font-size:13px; }}
main {{ padding:14px 20px 36px; max-width:1800px; margin:0 auto; }}
nav {{ position:sticky; top:0; z-index:10; display:flex; gap:6px; flex-wrap:wrap; margin:0 0 12px; padding:8px 0; background:var(--bg); border-bottom:1px solid var(--line); }}
nav button {{ background:#eef2f7; color:#27384f; border:1px solid #cbd4df; }}
nav button.active {{ background:var(--blue); color:white; border-color:var(--blue); }}
section {{ display:none; background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:14px; margin:0 0 14px; }}
section.active {{ display:block; }}
h2 {{ margin:0 0 12px; font-size:18px; line-height:1.25; }}
h3 {{ margin:18px 0 8px; font-size:15px; line-height:1.25; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:8px; }}
.card {{ border:1px solid var(--line); border-radius:6px; padding:10px; background:#fbfcfe; min-width:0; }}
.label {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
.value {{ font-size:14px; margin-top:4px; overflow-wrap:anywhere; }}
.toolbar {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:0 0 12px; }}
.toolbar label {{ color:var(--muted); font-size:13px; }}
select, input[type="search"] {{ border:1px solid var(--line); border-radius:6px; background:white; color:var(--ink); padding:8px 10px; min-height:34px; }}
.table-wrap {{ width:100%; overflow:auto; border:1px solid var(--line); border-radius:8px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ text-align:left; border-bottom:1px solid var(--line); padding:8px; vertical-align:top; }}
th {{ background:#eef2f7; position:sticky; top:0; z-index:1; }}
tbody tr:hover {{ background:#f8fbff; }}
textarea {{ width:100%; min-height:180px; box-sizing:border-box; font-family:Consolas, monospace; }}
button {{ background:var(--blue); color:white; border:0; border-radius:6px; padding:7px 10px; cursor:pointer; margin:6px 6px 0 0; min-height:32px; }}
button:hover {{ filter:brightness(.96); }}
pre {{ white-space:pre-wrap; word-break:break-word; background:#f7f8fa; border:1px solid var(--line); border-radius:6px; padding:8px; }}
.badge {{ display:inline-block; padding:2px 7px; border-radius:999px; background:#e8eef8; color:#123b73; font-size:12px; }}
.badge.error {{ background:var(--red); color:var(--red2); }}
.badge.ok {{ background:var(--green); color:var(--green2); padding:2px 7px; }}
.model-name {{ font-weight:600; }}
.replace {{ background:var(--red); color:var(--red2); border-radius:3px; padding:1px 3px; }}
.insert {{ background:var(--red); color:var(--red2); text-decoration:line-through; border-radius:3px; padding:1px 3px; }}
.delete {{ background:var(--red); color:var(--red2); border-radius:3px; padding:1px 3px; }}
.equal {{ background:transparent; }}
.diffline {{ line-height:1.9; font-family:Consolas, monospace; }}
.correction {{ font-size:12px; font-family:Segoe UI, Arial, sans-serif; color:var(--red2); margin-left:2px; }}
.pager {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:0 0 12px; }}
.ok {{ background:var(--green); padding:8px; border-radius:6px; }}
.warn {{ background:#fff4d6; padding:8px; border-radius:6px; }}
.chunk-row {{ border-top:1px solid var(--line); padding-top:10px; }}
.empty {{ color:var(--muted); font-style:italic; }}
.summary-table th {{ width:170px; }}
.path-cell {{ overflow-wrap:anywhere; }}
@media (max-width: 760px) {{
  header {{ padding:18px 16px; }}
  main {{ padding:14px 12px 32px; }}
  section {{ padding:14px; }}
  nav {{ gap:6px; }}
  nav button {{ padding:8px 10px; }}
}}
</style>
</head>
<body>
<header>
  <div><h1>Easy ASR Bench</h1>
  <p>Local transcription benchmark report</p></div>
</header>
<main>
  <nav>
    <button data-tab="overview" class="active" onclick="tab('overview')">Overview</button>
    <button data-tab="score" onclick="tab('score')">Scoreboard</button>
    <button data-tab="transcripts" onclick="tab('transcripts')">Transcripts</button>
    <button data-tab="chunks" onclick="tab('chunks')">Chunks</button>
    <button data-tab="pairwise" onclick="tab('pairwise')">Pairwise</button>
    <button data-tab="raw" onclick="tab('raw')">Raw</button>
    <button data-tab="export" onclick="tab('export')">Export</button>
  </nav>
  <section id="overview" class="active"></section>
  <section id="score">
    <h2>Reference and Accuracy</h2>
    <p>Scores compare each transcript with the corrected reference shown below. If no reference is loaded, this page can only rank speed and memory.</p>
    <div id="precomputedReferencePanel"></div>
    <textarea id="referenceInput" placeholder="Paste easy_asr_bench.llm_reference.v1 JSON here"></textarea>
    <button onclick="scoreReference()">Validate Reference and Score Models</button>
    <button onclick="downloadScored()">Download scored_report.json</button>
    <div id="referenceStatus"></div>
    <h2>Model Results</h2><div id="scoreboard"></div>
    <h2>Speed and Memory Only</h2><div id="runtimeRanking"></div>
  </section>
  <section id="pairwise"><h2>Pairwise Differences</h2><div id="pairwiseBody"></div></section>
  <section id="transcripts"><h2>Transcript Comparison</h2><div id="transcriptsBody"></div></section>
  <section id="chunks"><h2>Chunk Explorer</h2><div id="chunksBody"></div></section>
  <section id="raw"><h2>Raw Results JSON</h2><pre id="rawBody"></pre></section>
  <section id="export"><h2>Export</h2><button onclick="downloadScored()">Download scored_report.json</button><button onclick="copyBestTranscript()">Copy best transcript</button><div id="exportStatus"></div></section>
</main>
<script type="application/json" id="results-json">{embedded}</script>
<script>
const results = JSON.parse(document.getElementById('results-json').textContent);
let scored = null;
let latestScores = null;
const precomputedReferenceScores = results.reference_scores || null;
const precomputedReference = results.reference || null;
let chunkPage = 0;
let transcriptModelFilter = 'all';
let chunkModelFilter = 'all';
const pageSize = 25;
const transcriptPageSize = 5000;
const alignmentPageSize = 500;
const maxBrowserScoreCells = 12000000;
const transcriptPages = {{}};
const alignmentPages = {{}};
function tab(id) {{
  for (const s of document.querySelectorAll('section')) s.classList.toggle('active', s.id===id);
  for (const b of document.querySelectorAll('nav button')) b.classList.toggle('active', b.dataset.tab===id);
}}
function words(s, normalized=true) {{
  s = String(s || '');
  if (normalized) s = s.normalize('NFKC').toLocaleLowerCase().replace(/[^\\p{{L}}\\p{{N}}\\p{{M}}'\\s]+/gu, ' ');
  return s.trim().replace(/\\s+/g, ' ').split(' ').filter(Boolean);
}}
function edit(a,b) {{
  const dp = Array(a.length+1).fill(null).map(()=>Array(b.length+1).fill(0));
  for (let i=0;i<=a.length;i++) dp[i][0]=i;
  for (let j=0;j<=b.length;j++) dp[0][j]=j;
  for (let i=1;i<=a.length;i++) for (let j=1;j<=b.length;j++) {{
    dp[i][j]=Math.min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(a[i-1]===b[j-1]?0:1));
  }}
  return dp[a.length][b.length];
}}
function editCounts(a,b) {{
  const dp = Array(a.length+1).fill(null).map(()=>Array(b.length+1).fill(0));
  const op = Array(a.length+1).fill(null).map(()=>Array(b.length+1).fill(''));
  for (let i=1;i<=a.length;i++) {{ dp[i][0]=i; op[i][0]='delete'; }}
  for (let j=1;j<=b.length;j++) {{ dp[0][j]=j; op[0][j]='insert'; }}
  for (let i=1;i<=a.length;i++) for (let j=1;j<=b.length;j++) {{
    const choices = a[i-1]===b[j-1] ? [[dp[i-1][j-1], 'equal']] : [[dp[i-1][j-1]+1, 'replace']];
    choices.push([dp[i-1][j]+1, 'delete'], [dp[i][j-1]+1, 'insert']);
    choices.sort((x,y)=>x[0]-y[0]);
    dp[i][j]=choices[0][0]; op[i][j]=choices[0][1];
  }}
  let i=a.length, j=b.length, substitutions=0, insertions=0, deletions=0;
  while (i>0 || j>0) {{
    const action = op[i][j];
    if (action === 'equal') {{ i--; j--; }}
    else if (action === 'replace') {{ substitutions++; i--; j--; }}
    else if (action === 'delete') {{ deletions++; i--; }}
    else {{ insertions++; j--; }}
  }}
  return {{substitutions, insertions, deletions, edits: dp[a.length][b.length]}};
}}
function alignment(a,b) {{
  const dp = Array(a.length+1).fill(null).map(()=>Array(b.length+1).fill(0));
  const op = Array(a.length+1).fill(null).map(()=>Array(b.length+1).fill(''));
  for (let i=1;i<=a.length;i++) {{ dp[i][0]=i; op[i][0]='delete'; }}
  for (let j=1;j<=b.length;j++) {{ dp[0][j]=j; op[0][j]='insert'; }}
  for (let i=1;i<=a.length;i++) for (let j=1;j<=b.length;j++) {{
    const choices = a[i-1]===b[j-1] ? [[dp[i-1][j-1], 'equal']] : [[dp[i-1][j-1]+1, 'replace']];
    choices.push([dp[i-1][j]+1, 'delete'], [dp[i][j-1]+1, 'insert']);
    choices.sort((x,y)=>x[0]-y[0]);
    dp[i][j]=choices[0][0]; op[i][j]=choices[0][1];
  }}
  let i=a.length, j=b.length, out=[];
  while (i>0 || j>0) {{
    const action = op[i][j];
    if (action === 'equal') {{ out.push({{op:'equal', reference:a[i-1], hypothesis:b[j-1]}}); i--; j--; }}
    else if (action === 'replace') {{ out.push({{op:'replace', reference:a[i-1], hypothesis:b[j-1]}}); i--; j--; }}
    else if (action === 'delete') {{ out.push({{op:'delete', reference:a[i-1], hypothesis:''}}); i--; }}
    else {{ out.push({{op:'insert', reference:'', hypothesis:b[j-1]}}); j--; }}
  }}
  return out.reverse();
}}
function balancedScore(score, metrics) {{
  if (Number.isFinite(Number(score?.balanced_score))) return Number(score.balanced_score);
  if (!Number.isFinite(score?.normalized_wer)) return NaN;
  const quality = Math.max(0, 1 - score.normalized_wer);
  const speed = Math.min(1, Math.max(0, Number(metrics?.audio_seconds_per_wall_second || 0) / 10));
  const ram = Number(metrics?.peak_process_memory_mb || 0);
  const memory = ram > 0 ? Math.max(0, Math.min(1, 1 - ram / 32768)) : 0;
  return Math.max(0, Math.min(1, 0.70 * quality + 0.20 * speed + 0.10 * memory));
}}
function wer(ref,hyp,norm=true) {{ const r=words(ref,norm), h=words(hyp,norm); return edit(r,h)/Math.max(1,r.length); }}
function fullText(run) {{ return (run.transcript_chunks || []).map(c=>c.text || '').join('\\n'); }}
function fmt(n) {{ return Number.isFinite(n) ? n.toFixed(3) : 'n/a'; }}
function fmtSeconds(n) {{ const value = Number(n); return Number.isFinite(value) ? `${{value.toFixed(2)}} seconds` : 'not available'; }}
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
function renderOverview() {{
  const source = results.source || {{}};
  const runs = results.runs || [];
  const fastest = [...runs].sort((a,b)=>(b.metrics?.audio_seconds_per_wall_second||0)-(a.metrics?.audio_seconds_per_wall_second||0))[0];
  const lowRam = [...runs].sort((a,b)=>(a.metrics?.peak_process_memory_mb||Infinity)-(b.metrics?.peak_process_memory_mb||Infinity))[0];
  const errors = results.errors || [];
  const errorPanel = errors.length ? `<h3>Run Status</h3>${{errors.map(error => {{
    if (typeof error === 'string') return `<div class="warn">${{safe(error)}}</div>`;
    const causes = (error.likely_causes || []).map(item => `<li>${{safe(item)}}</li>`).join('');
    const actions = (error.next_actions || []).map(item => `<li>${{safe(item)}}</li>`).join('');
    return `<div class="warn"><b>${{safe(error.status || 'failed')}}</b><br>Stage: ${{safe(error.stage || 'unknown')}}<br>Problem: ${{safe(error.message || '')}}<h3>Likely causes</h3><ul>${{causes}}</ul><h3>Next actions</h3><ul>${{actions}}</ul><p>No source files were modified.</p></div>`;
  }}).join('')}}` : '';
  const referenceWarning = precomputedReference ? `<h3>Reference Status</h3><div class="warn">${{safe(results.score_note || 'LLM-corrected reference scores are not human ground truth.')}} ${{referenceUncertaintyText(precomputedReference)}}</div>` : '';
  document.getElementById('overview').innerHTML = `<h2>Run Summary</h2><div class="table-wrap"><table class="summary-table"><tbody>
    <tr><th>Source file</th><td class="path-cell">${{safe(source.name || '')}}</td></tr>
    <tr><th>Duration</th><td>${{fmtSeconds(source.duration_seconds || 0)}}</td></tr>
    <tr><th>Models tested</th><td>${{runs.length}}</td></tr>
    <tr><th>Audio chunks</th><td>${{(results.chunk_plan?.chunks || []).length}}</td></tr>
    <tr><th>Fastest model</th><td>${{safe(fastest?.model?.display_name || 'n/a')}}</td></tr>
    <tr><th>Lowest RAM model</th><td>${{safe(lowRam?.model?.display_name || 'n/a')}}</td></tr>
  </tbody></table></div>${{referenceWarning}}${{errorPanel}}`;
}}
function safe(s) {{ return escapeHtml(s); }}
function renderScoreboard(scores=null) {{
  const hasScores = !!scores;
  const rows = (results.runs || []).map(run => {{
    const m = run.metrics || {{}};
    const s = scores?.[run.model.candidate_id] || {{}};
    const errorBadge = run.errors?.length ? `<span class="badge error">${{run.errors.length}} runtime issue(s)</span>` : '<span class="badge ok">0 runtime issues</span>';
    const vramSource = m.vram_measurement_source || 'unavailable';
    const rank = s.balanced_rank ? `#${{safe(s.balanced_rank)}}` : 'not scored';
    const scoreText = value => hasScores ? fmt(value) : 'needs reference';
    const scorePercentText = value => hasScores ? fmtPercent(value) : 'needs reference';
    const countText = value => hasScores ? (value ?? 'n/a') : 'needs reference';
    return `<tr><td><span class="model-name">${{safe(run.model.display_name)}}</span><br><span class="badge">${{safe(run.model.precision)}}</span> ${{errorBadge}}</td><td>${{safe(run.model.backend)}}</td><td>${{rank}}</td><td>${{scorePercentText(s.normalized_wer)}}</td><td>${{scorePercentText(s.strict_wer)}}</td><td>${{scorePercentText(s.cer)}}</td><td>${{countText(s.substitutions)}}</td><td>${{countText(s.insertions)}}</td><td>${{countText(s.deletions)}}</td><td>${{scoreText(balancedScore(s,m))}}</td><td>${{fmtSeconds(m.total_wall_seconds)}}</td><td>${{fmtMemoryMb(m.peak_process_memory_mb)}}</td><td>${{fmtMemoryMb(m.peak_vram_mb)}}<br><span class="badge">${{safe(vramSource)}}</span></td></tr>`;
  }}).join('');
  const note = hasScores ? 'Accuracy scores use the corrected reference shown on this page. Lower word/character error is better; higher balanced score is better.' : 'Accuracy is not scored yet because no corrected reference is loaded. Speed and memory are still shown.';
  document.getElementById('scoreboard').innerHTML = `<div class="table-wrap"><table><thead><tr><th>Model</th><th>Backend</th><th>Accuracy Rank</th><th>Word Error Rate</th><th>Exact Word Error Rate</th><th>Character Error Rate</th><th>Changed Words</th><th>Extra Words</th><th>Missing Words</th><th>Balanced Score</th><th>Transcription Time</th><th>System RAM Peak</th><th>VRAM / GPU Memory Peak</th></tr></thead><tbody>${{rows}}</tbody></table></div><p class="empty">${{safe(note)}} VRAM/GPU memory is reported separately from process RAM. On integrated GPUs it can be shared system memory, so do not add it to RAM as a total.</p>`;
  renderRuntimeRanking();
}}
function renderRuntimeRanking() {{
  const ranking = results.runtime_rankings?.rows || [];
  if (!ranking.length) {{ document.getElementById('runtimeRanking').innerHTML = '<p class="empty">No runtime ranking is available.</p>'; return; }}
  const rows = ranking.map(row => `<tr><td>#${{safe(row.runtime_rank)}}</td><td>${{safe(row.display_name)}}</td><td>${{fmt(row.speed_percentile)}}</td><td>${{fmt(row.memory_percentile_inverse)}}</td><td>${{fmt(row.speed_audio_seconds_per_wall_second)}}x</td><td>${{fmtMemoryMb(row.peak_process_memory_mb)}}</td><td>${{fmtMemoryMb(row.peak_vram_mb)}}</td></tr>`).join('');
  document.getElementById('runtimeRanking').innerHTML = `<div class="table-wrap"><table><thead><tr><th>Rank</th><th>Model</th><th>Speed Percentile</th><th>Memory Percentile</th><th>Relative Speed</th><th>System RAM Peak</th><th>VRAM / GPU Memory Peak</th></tr></thead><tbody>${{rows}}</tbody></table></div><p class="empty">${{safe(results.runtime_rankings?.note || 'Runtime ranking does not measure transcript quality.')}} Relative speed is audio duration divided by run time; the main scoreboard shows transcription time in seconds.</p>`;
}}
function referenceUncertaintyText(ref) {{
  const notes = [];
  for (const segment of ref?.segments || []) {{
    for (const item of segment.uncertain || []) notes.push(item);
  }}
  for (const item of ref?.global_notes || []) notes.push(item);
  const unique = [...new Set(notes.filter(Boolean))];
  return unique.length ? ` Notes: ${{unique.map(safe).join('; ')}}` : '';
}}
function referenceFullText(ref) {{
  const chunks = results.chunk_plan?.chunks || [];
  const refMap = new Map((ref?.segments || []).map(segment => [segment.chunk_id, segment]));
  return chunks.map(chunk => refMap.get(chunk.chunk_id)?.text || '').join('\\n');
}}
function renderPrecomputedReferencePanel(ref, scores) {{
  const panel = document.getElementById('precomputedReferencePanel');
  if (!panel) return;
  if (!ref) {{
    panel.innerHTML = '';
    return;
  }}
  const segments = (ref.segments || []).map(segment => `<tr><td>${{safe(segment.chunk_id)}}</td><td>${{fmt(Number(segment.start_seconds))}}-${{fmt(Number(segment.end_seconds))}}s</td><td><pre>${{escapeHtml(segment.text || '')}}</pre></td><td>${{safe((segment.uncertain || []).join('; ') || 'none')}}</td></tr>`).join('');
  const scoreCount = scores ? Object.keys(scores).length : 0;
  panel.innerHTML = `<div class="card"><h3>Loaded LLM-Corrected Reference</h3><p><b>Score type:</b> ${{safe(results.score_type || 'llm_corrected_reference')}}. <b>Scored models:</b> ${{scoreCount}}. This is AI-assisted reference text, not human ground truth.</p><p>${{safe(referenceUncertaintyText(ref) || 'No uncertainty notes provided.')}}</p><h3>Reference Text</h3><pre>${{escapeHtml(referenceFullText(ref))}}</pre><div class="table-wrap"><table><thead><tr><th>Chunk</th><th>Time</th><th>Corrected Reference</th><th>Notes</th></tr></thead><tbody>${{segments}}</tbody></table></div></div>`;
}}
function renderPairwise() {{
  const rows = Object.entries(results.pairwise_differences || {{}}).map(([k,v]) => `<tr><td>${{safe(k)}}</td><td>${{fmt(v.normalized_wer_like_difference)}}</td><td>${{fmt(v.cer_difference)}}</td></tr>`).join('');
  document.getElementById('pairwiseBody').innerHTML = rows ? `<div class="table-wrap"><table><thead><tr><th>Pair</th><th>Norm WER-like Difference</th><th>CER Difference</th></tr></thead><tbody>${{rows}}</tbody></table></div>` : '<p class="empty">No pairwise differences available.</p>';
}}
function runOptions(selected) {{
  return '<option value="all">All models</option>' + (results.runs || []).map(run => `<option value="${{safe(run.model.candidate_id)}}" ${{selected===run.model.candidate_id?'selected':''}}>${{safe(run.model.display_name)}}</option>`).join('');
}}
function renderRunErrors(run) {{
  const errors = run.errors || [];
  if (!errors.length) return '';
  const body = errors.map(error => {{
    if (typeof error === 'string') return `<div class="warn">${{safe(error)}}</div>`;
    const causes = (error.likely_causes || []).map(item => `<li>${{safe(item)}}</li>`).join('');
    const actions = (error.next_actions || []).map(item => `<li>${{safe(item)}}</li>`).join('');
    const repair = error.repair_command ? `<p><b>Repair:</b> <code>${{safe(error.repair_command)}}</code></p>` : '';
    const log = error.log_path ? `<p><b>Log:</b> ${{safe(error.log_path)}}</p>` : '';
    return `<div class="warn"><b>Model failure</b><br>Stage: ${{safe(error.stage || 'unknown')}}<br>Problem: ${{safe(error.message || '')}}<h3>Likely causes</h3><ul>${{causes}}</ul><h3>Next actions</h3><ul>${{actions}}</ul>${{repair}}${{log}}</div>`;
  }}).join('');
  return `<h3>Model Errors</h3>${{body}}`;
}}
function renderTranscripts() {{
  const visibleRuns = (results.runs || []).filter(run => transcriptModelFilter === 'all' || run.model.candidate_id === transcriptModelFilter);
  const controls = `<div class="toolbar"><label for="transcriptModel">Model</label><select id="transcriptModel" onchange="transcriptModelFilter=this.value;renderTranscripts()">${{runOptions(transcriptModelFilter)}}</select></div>`;
  document.getElementById('transcriptsBody').innerHTML = controls + visibleRuns.map(run => {{
    const id = run.model.candidate_id;
    const s = latestScores?.[id];
    const body = s?.alignment ? renderAlignmentPage(id, s.alignment) : renderTranscriptTextPage(id, fullText(run));
    return `<h3>${{safe(run.model.display_name)}} <span class="badge">${{safe(run.model.precision)}}</span></h3>${{renderRunErrors(run)}}${{body}}`;
  }}).join('') || controls + '<p class="empty">No transcript is available for this filter.</p>';
}}
function renderTranscriptTextPage(runId, text) {{
  const pages = Math.max(1, Math.ceil(String(text).length / transcriptPageSize));
  const page = Math.min(transcriptPages[runId] || 0, pages - 1);
  transcriptPages[runId] = page;
  const start = page * transcriptPageSize;
  const slice = String(text).slice(start, start + transcriptPageSize);
  const controls = `<div class="pager"><button onclick="transcriptPages['${{safe(runId)}}']=Math.max(0,(${{page}})-1);renderTranscripts()">Prev</button><span>Transcript page ${{page+1}} / ${{pages}}</span><button onclick="transcriptPages['${{safe(runId)}}']=Math.min(${{pages-1}},(${{page}})+1);renderTranscripts()">Next</button></div>`;
  return `${{controls}}<pre>${{escapeHtml(slice)}}</pre>`;
}}
function renderAlignmentPage(runId, items) {{
  const pages = Math.max(1, Math.ceil((items || []).length / alignmentPageSize));
  const page = Math.min(alignmentPages[runId] || 0, pages - 1);
  alignmentPages[runId] = page;
  const slice = (items || []).slice(page * alignmentPageSize, (page + 1) * alignmentPageSize);
  const controls = `<div class="pager"><button onclick="alignmentPages['${{safe(runId)}}']=Math.max(0,(${{page}})-1);renderTranscripts()">Prev</button><span>Alignment page ${{page+1}} / ${{pages}}</span><button onclick="alignmentPages['${{safe(runId)}}']=Math.min(${{pages-1}},(${{page}})+1);renderTranscripts()">Next</button></div>`;
  return `${{controls}}<div class="diffline">${{renderAlignment(slice)}}</div>`;
}}
function renderChunks() {{
  const chunks = results.chunk_plan?.chunks || [];
  const totalPages = Math.max(1, Math.ceil(chunks.length / pageSize));
  chunkPage = Math.min(chunkPage, totalPages - 1);
  const pageChunks = chunks.slice(chunkPage * pageSize, (chunkPage + 1) * pageSize);
  const visibleRuns = (results.runs || []).filter(run => chunkModelFilter === 'all' || run.model.candidate_id === chunkModelFilter);
  const controls = `<div class="toolbar"><label for="chunkModel">Model</label><select id="chunkModel" onchange="chunkModelFilter=this.value;renderChunks()">${{runOptions(chunkModelFilter)}}</select></div>`;
  const pager = `<div class="pager"><button onclick="chunkPage=Math.max(0,chunkPage-1);renderChunks()">Prev</button><span>Chunk page ${{chunkPage+1}} / ${{totalPages}}</span><button onclick="chunkPage=Math.min(${{totalPages-1}},chunkPage+1);renderChunks()">Next</button></div>`;
  document.getElementById('chunksBody').innerHTML = controls + pager + pageChunks.map(c => `<div class="chunk-row"><h3>${{safe(c.chunk_id)}} [${{safe(c.start_timestamp)}} - ${{safe(c.end_timestamp)}}]</h3>` + visibleRuns.map(run => {{
    const t = (run.transcript_chunks||[]).find(x=>x.chunk_id===c.chunk_id);
    return `<div class="card"><b>${{safe(run.model.display_name)}}</b><pre>${{escapeHtml(t?.text || '')}}</pre></div>`;
  }}).join('') + '</div>').join('') + pager;
}}
function escapeHtml(s) {{ return String(s).replace(/[&<>]/g, c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c])); }}
function renderAlignment(items) {{
  return items.map(item => {{
    if (item.op === 'equal') return `<span class="equal">${{safe(item.hypothesis)}}</span>`;
    if (item.op === 'insert') return `<span class="insert">${{safe(item.hypothesis)}}</span>`;
    if (item.op === 'delete') return `<span class="delete">${{safe(item.reference)}}<span class="correction">missing</span></span>`;
    return `<span class="replace">${{safe(item.hypothesis)}}<span class="correction">-&gt; ${{safe(item.reference)}}</span></span>`;
  }}).join(' ');
}}
function extractJson(text) {{
  text = String(text || '').trim();
  const fence = text.match(/```(?:json)?\\s*([\\s\\S]*?)```/i);
  if (fence) text = fence[1].trim();
  if (!text.startsWith('{{')) {{
    const start = text.indexOf('{{'), end = text.lastIndexOf('}}');
    if (start >= 0 && end > start) text = text.slice(start, end+1);
  }}
  return JSON.parse(text);
}}
function scoreReference() {{
  let ref;
  try {{ ref = extractJson(document.getElementById('referenceInput').value); }}
  catch(e) {{ document.getElementById('referenceStatus').innerHTML = '<p class="warn">Reference JSON could not be parsed.</p>'; return; }}
  const expected = new Set((results.chunk_plan?.chunks || []).map(c=>c.chunk_id));
  const seen = new Set();
  const duplicates = [];
  const refMap = new Map();
  for (const segment of ref.segments || []) {{
    if (seen.has(segment.chunk_id)) duplicates.push(segment.chunk_id);
    seen.add(segment.chunk_id);
    refMap.set(segment.chunk_id, segment);
  }}
  const missing = [...expected].filter(id=>!refMap.has(id));
  const extra = [...refMap.keys()].filter(id=>!expected.has(id));
  const sourceMismatch = ref.source_sha256 && results.source?.sha256 && ref.source_sha256 !== results.source.sha256;
  const timestampErrors = [];
  for (const chunk of results.chunk_plan?.chunks || []) {{
    const segment = refMap.get(chunk.chunk_id);
    if (!segment) continue;
    if (Math.abs(Number(segment.start_seconds) - Number(chunk.start_seconds)) > 0.01 || Math.abs(Number(segment.end_seconds) - Number(chunk.end_seconds)) > 0.01) timestampErrors.push(chunk.chunk_id);
  }}
  if (ref.schema !== 'easy_asr_bench.llm_reference.v1' || ref.reference_type !== 'llm_corrected_reference' || missing.length || extra.length || duplicates.length || sourceMismatch || timestampErrors.length) {{
    document.getElementById('referenceStatus').innerHTML = `<p class="warn">Invalid reference. Missing chunks: ${{safe(missing.join(', ') || 'none')}}. Extra chunks: ${{safe(extra.join(', ') || 'none')}}. Duplicate chunks: ${{safe(duplicates.join(', ') || 'none')}}. Timestamp mismatches: ${{safe(timestampErrors.join(', ') || 'none')}}. Source hash mismatch: ${{sourceMismatch ? 'yes' : 'no'}}.</p>`;
    return;
  }}
  const referenceText = (results.chunk_plan?.chunks || []).map(c=>refMap.get(c.chunk_id)?.text || '').join('\\n');
  const refWordsForGuard = words(referenceText, true);
  const scoringTooLarge = (results.runs || []).some(run => refWordsForGuard.length * Math.max(1, words(fullText(run), true).length) > maxBrowserScoreCells);
  if (scoringTooLarge) {{
    document.getElementById('referenceStatus').innerHTML = `<p class="warn">Reference is valid, but this report is too large for browser WER/CER scoring without freezing. Use the raw results and LLM reference JSON with an offline/Python scorer, or rerun smaller batches.</p>`;
    return;
  }}
  const scores = {{}};
  for (const run of results.runs || []) {{
    const hyp = fullText(run);
    const refWords = refWordsForGuard;
    const hypWords = words(hyp, true);
    const counts = editCounts(refWords, hypWords);
    scores[run.model.candidate_id] = {{ normalized_wer: counts.edits/Math.max(1,refWords.length), strict_wer: wer(referenceText,hyp,false), cer: edit(referenceText,hyp)/Math.max(1,referenceText.length), substitutions: counts.substitutions, insertions: counts.insertions, deletions: counts.deletions, alignment: alignment(refWords, hypWords) }};
  }}
  scored = {{ results, reference: ref, scores }};
  latestScores = scores;
  document.getElementById('referenceStatus').innerHTML = '<p class="ok">Reference validated. Scores updated. Label these as LLM-corrected reference scores, not human ground truth.</p>';
  renderPrecomputedReferencePanel(ref, scores);
  renderScoreboard(scores);
  renderTranscripts();
}}
function downloadScored() {{
  const blob = new Blob([JSON.stringify(scored || {{results}}, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'scored_report.json'; a.click();
}}
if (precomputedReferenceScores) {{
  latestScores = precomputedReferenceScores;
  scored = {{ results, reference: precomputedReference, scores: precomputedReferenceScores, score_type: results.score_type || 'llm_corrected_reference', score_note: results.score_note }};
  document.getElementById('referenceStatus').innerHTML = '<p class="ok">Loaded precomputed LLM-corrected reference scores. These are not human ground truth.</p>';
  renderPrecomputedReferencePanel(precomputedReference, precomputedReferenceScores);
}}
renderOverview(); renderScoreboard(latestScores); renderPairwise(); renderTranscripts();
renderChunks();
function copyBestTranscript() {{
  const runs = results.runs || [];
  let best = runs[0];
  if (latestScores) best = [...runs].sort((a,b)=>(latestScores[a.model.candidate_id]?.normalized_wer??99)-(latestScores[b.model.candidate_id]?.normalized_wer??99))[0];
  navigator.clipboard.writeText(fullText(best || {{}}));
  document.getElementById('exportStatus').textContent = 'Best transcript copied.';
}}
document.getElementById('rawBody').textContent = JSON.stringify(results, null, 2);
</script>
</body>
</html>"""
