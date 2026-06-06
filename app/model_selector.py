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


def choose_candidates(candidates: list[ModelCandidate], unsupported: list[ModelCandidate]) -> list[ModelCandidate]:
    print("ASR Model Bench")
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
        print()
        print("Detected non-ASR, incomplete, or unsupported candidates:")
        print()
        for index, candidate in enumerate(unsupported, 1):
            reason = "; ".join(candidate.warnings + ([f"Missing: {', '.join(candidate.missing_files)}"] if candidate.missing_files else []))
            print(f"[U{index}] {candidate.display_name}")
            print(f"     Reason: {reason or 'Unsupported by current adapters.'}")
    if not candidates:
        return []
    print()
    print("Choose models: numbers like 1 2 4, 1,2,4, 1-4, A for all, R for recommended.")
    while True:
        raw = input("Models> ").strip()
        if raw.lower() in {"r", "recommended"}:
            indexes = recommended_candidates(candidates)
        else:
            indexes = parse_selection(raw, len(candidates))
        if indexes:
            break
        print("No valid runnable models selected.")
    selected = [candidates[index - 1] for index in indexes]
    return choose_precision_buckets(selected)


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
