from __future__ import annotations

import re

from .adapters.base import ModelCandidate


def parse_selection(raw: str, max_index: int) -> list[int]:
    text = raw.strip().lower()
    if text in {"a", "all"}:
        return list(range(1, max_index + 1))
    if not text:
        return []
    selected: set[int] = set()
    if re.fullmatch(r"\d+", text) and max_index <= 9 and len(text) > 1:
        selected.update(int(char) for char in text)
    elif re.fullmatch(r"\d+", text) and max_index > 9 and len(text) > 1:
        return []
    else:
        for part in re.split(r"[\s,]+", text):
            if not part:
                continue
            if "-" in part:
                left, right = part.split("-", 1)
                if left.isdigit() and right.isdigit():
                    start, end = int(left), int(right)
                    if start <= end:
                        selected.update(range(start, end + 1))
            elif part.isdigit():
                selected.add(int(part))
    return [index for index in sorted(selected) if 1 <= index <= max_index]


def recommended_candidates(candidates: list[ModelCandidate]) -> list[int]:
    indexes: list[int] = []
    wanted = {"int8", "fp16w"}
    for index, candidate in enumerate(candidates, 1):
        if candidate.precision in wanted and "granite" in candidate.adapter_name:
            indexes.append(index)
    return indexes or list(range(1, len(candidates) + 1))


def choose_candidates(candidates: list[ModelCandidate], unsupported: list[ModelCandidate]) -> tuple[list[ModelCandidate], ModelCandidate | None]:
    candidates = [candidate for candidate in candidates if candidate.category == "asr"]
    reference_llms = [candidate for candidate in unsupported if candidate.category == "reference_llm"]
    unsupported = [candidate for candidate in unsupported if candidate.category != "reference_llm"]
    print("Easy ASR Bench")
    print()
    print("Detected runnable ASR model variants:")
    print()
    if not candidates:
        print("  None")
    for index, candidate in enumerate(candidates, 1):
        print(
            f"[{index}] {candidate.display_name:<30} backend: {candidate.backend:<12} "
            f"precision: {candidate.precision:<8} bucket: {candidate.quantization_label}"
        )
        print(f"    Path: {candidate.path}")
    if unsupported:
        if reference_llms:
            print()
            print("Detected reference/correction LLMs:")
            print()
            for index, candidate in enumerate(reference_llms, 1):
                print(f"[L{index}] {candidate.display_name}    {candidate.backend}    {candidate.precision}    {candidate.quantization_label}")
                print(f"     Use: optional LLM-corrected reference generation, not direct transcription.")
        print()
        print("Detected non-ASR, incomplete, or unsupported candidates:")
        print()
        for index, candidate in enumerate(unsupported, 1):
            reason = "; ".join(candidate.warnings + ([f"Missing: {', '.join(candidate.missing_files)}"] if candidate.missing_files else []))
            print(f"[U{index}] {candidate.display_name}")
            print(f"     Reason: {reason or 'Unsupported by current adapters.'}")
    if not candidates:
        return [], None
    print()
    if len(candidates) > 9:
        print("Choose models with spaces, commas, or ranges: 1 2 10, 1,2,10, 1-4. Compact digits are disabled for 10+ models.")
    else:
        print("Choose models: numbers like 1 2 4, 1,2,4, 1-4, 1234, A for all, R for recommended.")
    while True:
        raw = input("Models> ").strip()
        if raw.lower() in {"r", "recommended"}:
            indexes = recommended_candidates(candidates)
        else:
            indexes = parse_selection(raw, len(candidates))
        if indexes:
            break
        print("No valid runnable models selected.")
    selected = choose_precision_buckets([candidates[index - 1] for index in indexes])
    reference_llm = choose_reference_llm(reference_llms)
    return selected, reference_llm


def choose_precision_buckets(candidates: list[ModelCandidate]) -> list[ModelCandidate]:
    buckets = sorted({candidate.quantization_label for candidate in candidates})
    print()
    print("Available precision buckets in selected models:")
    print()
    for index, bucket in enumerate(buckets, 1):
        print(f"[{index}] {bucket}")
    print("[A] All available precisions")
    while True:
        raw = input("Precisions> ").strip()
        if raw.lower() in {"a", "all", ""}:
            return candidates
        indexes = parse_selection(raw, len(buckets))
        if indexes:
            chosen = {buckets[index - 1] for index in indexes}
            return [candidate for candidate in candidates if candidate.quantization_label in chosen]
        print("No valid precision buckets selected.")


def choose_reference_llm(reference_llms: list[ModelCandidate]) -> ModelCandidate | None:
    if not reference_llms:
        return None
    print()
    answer = input("Use a local GGUF LLM for LLM-corrected reference help? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        return None
    for index, candidate in enumerate(reference_llms, 1):
        print(f"[{index}] {candidate.display_name} | {candidate.precision} | {candidate.path}")
    while True:
        raw = input("Reference LLM> ").strip()
        indexes = parse_selection(raw, len(reference_llms))
        if len(indexes) == 1:
            return reference_llms[indexes[0] - 1]
        print("Choose one reference LLM number, or press Ctrl+C to skip.")
