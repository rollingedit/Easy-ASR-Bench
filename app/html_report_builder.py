from __future__ import annotations

import html
import json


def build_html_report(results: dict) -> str:
    embedded = html.escape(json.dumps(results, ensure_ascii=False))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Easy ASR Bench Report</title>
<style>
:root {{ color-scheme: light; --bg:#f6f7f9; --panel:#fff; --ink:#17202a; --muted:#5f6b7a; --line:#d9dee7; --blue:#174ea6; --red:#ffd7d7; --yellow:#fff0b3; --green:#dff4e4; }}
body {{ margin:0; font-family:Segoe UI, Arial, sans-serif; background:var(--bg); color:var(--ink); }}
header {{ padding:24px 32px; background:#0f1720; color:white; }}
header h1 {{ margin:0 0 6px; font-size:26px; }}
header p {{ margin:0; color:#c8d1dc; }}
main {{ padding:24px 32px 48px; max-width:1400px; margin:0 auto; }}
nav {{ display:flex; gap:8px; flex-wrap:wrap; margin:0 0 18px; }}
nav button {{ background:#e8eef8; color:#123b73; }}
section {{ display:none; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; margin:0 0 18px; }}
section.active {{ display:block; }}
h2 {{ margin:0 0 12px; font-size:18px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
.card {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:#fbfcfe; }}
.label {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
.value {{ font-size:20px; margin-top:4px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ text-align:left; border-bottom:1px solid var(--line); padding:8px; vertical-align:top; }}
th {{ background:#f0f3f7; position:sticky; top:0; }}
textarea {{ width:100%; min-height:180px; box-sizing:border-box; font-family:Consolas, monospace; }}
button {{ background:var(--blue); color:white; border:0; border-radius:6px; padding:9px 12px; cursor:pointer; margin:8px 8px 0 0; }}
pre {{ white-space:pre-wrap; word-break:break-word; background:#f7f8fa; border:1px solid var(--line); border-radius:6px; padding:10px; }}
.badge {{ display:inline-block; padding:2px 7px; border-radius:999px; background:#e8eef8; color:#123b73; font-size:12px; }}
.replace {{ background:var(--red); }}
.insert {{ background:var(--yellow); }}
.delete {{ text-decoration:line-through; background:var(--red); }}
.ok {{ background:var(--green); padding:8px; border-radius:6px; }}
.warn {{ background:#fff4d6; padding:8px; border-radius:6px; }}
</style>
</head>
<body>
<header>
  <h1>Easy ASR Bench Report</h1>
  <p>Local model comparison with transcripts, speed, memory, pairwise differences, and LLM-corrected reference scoring.</p>
</header>
<main>
  <nav>
    <button onclick="tab('overview')">Overview</button>
    <button onclick="tab('score')">Scoreboard</button>
    <button onclick="tab('transcripts')">Transcripts</button>
    <button onclick="tab('chunks')">Chunks</button>
    <button onclick="tab('pairwise')">Pairwise</button>
    <button onclick="tab('raw')">Raw</button>
    <button onclick="tab('export')">Export</button>
  </nav>
  <section id="overview" class="active"></section>
  <section id="score">
    <h2>LLM-Corrected Reference</h2>
    <p>Paste JSON returned by an LLM using the prompt in <code>results.txt</code>. This scores against an LLM-corrected reference, not human ground truth.</p>
    <textarea id="referenceInput" placeholder="Paste easy_asr_bench.llm_reference.v1 JSON here"></textarea>
    <button onclick="scoreReference()">Validate Reference and Score Models</button>
    <button onclick="downloadScored()">Download scored_report.json</button>
    <div id="referenceStatus"></div>
    <h2>Scoreboard</h2><div id="scoreboard"></div>
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
function tab(id) {{ for (const s of document.querySelectorAll('section')) s.classList.toggle('active', s.id===id); }}
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
function wer(ref,hyp,norm=true) {{ const r=words(ref,norm), h=words(hyp,norm); return edit(r,h)/Math.max(1,r.length); }}
function fullText(run) {{ return (run.transcript_chunks || []).map(c=>c.text || '').join('\\n'); }}
function fmt(n) {{ return Number.isFinite(n) ? n.toFixed(3) : 'n/a'; }}
function renderOverview() {{
  const source = results.source || {{}};
  const runs = results.runs || [];
  const fastest = [...runs].sort((a,b)=>(b.metrics?.audio_seconds_per_wall_second||0)-(a.metrics?.audio_seconds_per_wall_second||0))[0];
  const lowRam = [...runs].sort((a,b)=>(a.metrics?.peak_process_memory_mb||Infinity)-(b.metrics?.peak_process_memory_mb||Infinity))[0];
  document.getElementById('overview').innerHTML = `<h2>Overview</h2><div class="grid">
    <div class="card"><div class="label">Source</div><div class="value">${{safe(source.name || '')}}</div></div>
    <div class="card"><div class="label">Duration</div><div class="value">${{fmt(source.duration_seconds || 0)}}s</div></div>
    <div class="card"><div class="label">Models Tested</div><div class="value">${{runs.length}}</div></div>
    <div class="card"><div class="label">Chunks</div><div class="value">${{(results.chunk_plan?.chunks || []).length}}</div></div>
    <div class="card"><div class="label">Fastest</div><div class="value">${{safe(fastest?.model?.display_name || 'n/a')}}</div></div>
    <div class="card"><div class="label">Lowest RAM</div><div class="value">${{safe(lowRam?.model?.display_name || 'n/a')}}</div></div>
  </div>`;
}}
function safe(s) {{ return escapeHtml(s); }}
function renderScoreboard(scores=null) {{
  const rows = (results.runs || []).map(run => {{
    const m = run.metrics || {{}};
    const s = scores?.[run.model.candidate_id] || {{}};
    return `<tr><td>${{safe(run.model.display_name)}}<br><span class="badge">${{safe(run.model.precision)}}</span></td><td>${{safe(run.model.backend)}}</td><td>${{fmt(s.normalized_wer)}}</td><td>${{fmt(s.strict_wer)}}</td><td>${{fmt(s.cer)}}</td><td>${{s.substitutions ?? 'n/a'}}</td><td>${{s.insertions ?? 'n/a'}}</td><td>${{s.deletions ?? 'n/a'}}</td><td>${{fmt(m.audio_seconds_per_wall_second)}}</td><td>${{fmt(m.total_wall_seconds)}}</td><td>${{fmt(m.peak_process_memory_mb)}}</td><td>${{fmt(m.peak_vram_mb)}}</td><td>${{run.errors?.length || 0}}</td></tr>`;
  }}).join('');
  document.getElementById('scoreboard').innerHTML = `<table><thead><tr><th>Model</th><th>Backend</th><th>Norm WER</th><th>Strict WER</th><th>CER</th><th>Sub</th><th>Ins</th><th>Del</th><th>x Real-time</th><th>Wall s</th><th>RAM MB</th><th>VRAM MB</th><th>Errors</th></tr></thead><tbody>${{rows}}</tbody></table>`;
}}
function renderPairwise() {{
  const rows = Object.entries(results.pairwise_differences || {{}}).map(([k,v]) => `<tr><td>${{safe(k)}}</td><td>${{fmt(v.normalized_wer_like_difference)}}</td><td>${{fmt(v.cer_difference)}}</td></tr>`).join('');
  document.getElementById('pairwiseBody').innerHTML = rows ? `<table><thead><tr><th>Pair</th><th>Norm WER-like Difference</th><th>CER Difference</th></tr></thead><tbody>${{rows}}</tbody></table>` : '<p>No pairwise differences available.</p>';
}}
function renderTranscripts() {{
  document.getElementById('transcriptsBody').innerHTML = (results.runs || []).map(run => `<h3>${{safe(run.model.display_name)}} <span class="badge">${{safe(run.model.precision)}}</span></h3><pre>${{escapeHtml(fullText(run))}}</pre>`).join('');
}}
function renderChunks() {{
  const chunks = results.chunk_plan?.chunks || [];
  document.getElementById('chunksBody').innerHTML = chunks.map(c => `<h3>${{safe(c.chunk_id)}} [${{safe(c.start_timestamp)}} - ${{safe(c.end_timestamp)}}]</h3>` + (results.runs||[]).map(run => {{
    const t = (run.transcript_chunks||[]).find(x=>x.chunk_id===c.chunk_id);
    return `<div class="card"><b>${{safe(run.model.display_name)}}</b><pre>${{escapeHtml(t?.text || '')}}</pre></div>`;
  }}).join('')).join('');
}}
function escapeHtml(s) {{ return String(s).replace(/[&<>]/g, c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c])); }}
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
  if (ref.schema !== 'easy_asr_bench.llm_reference.v1' || ref.reference_type !== 'llm_corrected_reference' || missing.length || extra.length || duplicates.length || sourceMismatch) {{
    document.getElementById('referenceStatus').innerHTML = `<p class="warn">Invalid reference. Missing chunks: ${{safe(missing.join(', ') || 'none')}}. Extra chunks: ${{safe(extra.join(', ') || 'none')}}. Duplicate chunks: ${{safe(duplicates.join(', ') || 'none')}}. Source hash mismatch: ${{sourceMismatch ? 'yes' : 'no'}}.</p>`;
    return;
  }}
  const referenceText = (results.chunk_plan?.chunks || []).map(c=>refMap.get(c.chunk_id)?.text || '').join('\\n');
  const scores = {{}};
  for (const run of results.runs || []) {{
    const hyp = fullText(run);
    const refWords = words(referenceText, true);
    const hypWords = words(hyp, true);
    const counts = editCounts(refWords, hypWords);
    scores[run.model.candidate_id] = {{ normalized_wer: counts.edits/Math.max(1,refWords.length), strict_wer: wer(referenceText,hyp,false), cer: edit(referenceText,hyp)/Math.max(1,referenceText.length), substitutions: counts.substitutions, insertions: counts.insertions, deletions: counts.deletions }};
  }}
  scored = {{ results, reference: ref, scores }};
  latestScores = scores;
  document.getElementById('referenceStatus').innerHTML = '<p class="ok">Reference validated. Scores updated. Label these as LLM-corrected reference scores, not human ground truth.</p>';
  renderScoreboard(scores);
}}
function downloadScored() {{
  const blob = new Blob([JSON.stringify(scored || {{results}}, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'scored_report.json'; a.click();
}}
renderOverview(); renderScoreboard(); renderPairwise(); renderTranscripts();
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
