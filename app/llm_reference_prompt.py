from __future__ import annotations

import json


def build_llm_reference_prompt(results: dict) -> str:
    compact = {
        "schema": results["schema"],
        "source": results["source"],
        "chunk_plan": results["chunk_plan"],
        "runs": [
            {
                "model_id": run["model"]["candidate_id"],
                "display_name": run["model"]["display_name"],
                "chunks": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "start_seconds": chunk["start_seconds"],
                        "end_seconds": chunk["end_seconds"],
                        "text": chunk["text"],
                    }
                    for chunk in run.get("transcript_chunks", [])
                ],
            }
            for run in results.get("runs", [])
        ],
    }
    return (
        "You are creating an LLM-corrected reference transcript from multiple ASR outputs.\n"
        "Preserve spoken meaning. Do not summarize. Do not invent missing speech.\n"
        "Use one segment per original chunk. If the audio content is uncertain, put a short note in that segment's uncertain list.\n"
        "This is an LLM-corrected reference, not human ground truth.\n"
        "Return only valid JSON matching this schema:\n\n"
        "{\n"
        '  "schema": "easy_asr_bench.llm_reference.v1",\n'
        '  "source_sha256": "<copy source sha256 from results if present>",\n'
        '  "reference_type": "llm_corrected_reference",\n'
        '  "segments": [\n'
        '    {"chunk_id": "0001", "start_seconds": 0.0, "end_seconds": 10.0, "text": "corrected transcript", "uncertain": []}\n'
        "  ],\n"
        '  "global_notes": []\n'
        "}\n\n"
        "BEGIN_RESULTS_JSON\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}\n"
        "END_RESULTS_JSON\n"
    )
