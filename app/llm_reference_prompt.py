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
        "Your job is to infer the most likely spoken words from the agreement and disagreements between ASR models.\n"
        "You receive source metadata, the chunk plan, and every selected ASR model's transcript chunks in BEGIN_RESULTS_JSON.\n"
        "Compare the model outputs chunk by chunk. Use agreement between models as evidence, and use coherent wording over obvious garble.\n"
        "Output format rules:\n"
        "- Return only valid JSON. Do not include Markdown fences, commentary, headings, confidence scores, or a notes section outside JSON.\n"
        "- Keep exactly one JSON segment per original chunk_id.\n"
        "- Keep each segment's chunk_id, start_seconds, and end_seconds aligned to the input chunk.\n"
        "- Put the corrected transcript text for that chunk in the segment text field.\n"
        "- If speakers are clearly present and supported by ASR evidence, short speaker labels may be included inside text. Do not invent speaker labels.\n"
        "- Always choose the best-supported wording from the ASR outputs. Keep the segment's uncertain list empty unless the input already provides an explicit human note that must be preserved.\n"
        "Preserve the speaker's intent, topic, dialogue flow, tone, wording style, and register. The source may be a podcast, interview, lecture, news clip, comedy bit, meeting, dialogue scene, phone call, informal chat, scripted speech, technical discussion, or explicit/NSFW speech.\n"
        "Keep topic-specific terms, dialogue flow, names, acronyms, numbers, quoted phrases, filler words, dialect, slang, jokes, informal grammar, incomplete sentences, false starts, cut-off thoughts, and profanity when the ASR evidence supports them. Do not sanitize, formalize, summarize, paraphrase, complete fragments into polished grammar, or make the text sound more polished than the speaker.\n"
        "Use punctuation and capitalization to improve readability, but do not let punctuation edits change the spoken meaning.\n"
        "Prefer words supported by multiple ASR outputs. If one model is clearly garbled and another is coherent, use the coherent wording. If all models disagree, still choose the most plausible wording from the available ASR evidence; do not stop the correction with an uncertainty placeholder.\n"
        "Do not invent missing speech, facts, speaker labels, or context that is not supported by the ASR outputs.\n"
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
