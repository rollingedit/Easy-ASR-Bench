from __future__ import annotations

import re
from pathlib import Path

from .adapters.base import ModelCandidate
from .llm_reference import merge_reference_llms, print_external_llm_guide, save_custom_reference_path, scan_custom_reference_llms


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


def choose_candidates(
    candidates: list[ModelCandidate],
    unsupported: list[ModelCandidate],
    config: dict | None = None,
    config_path: Path | None = None,
) -> tuple[list[ModelCandidate], ModelCandidate | None]:
    candidates = [candidate for candidate in candidates if candidate.category == "asr"]
    saved_reference_llms = scan_custom_reference_llms(config) if config is not None else []
    reference_llms = merge_reference_llms([candidate for candidate in unsupported if candidate.category == "reference_llm"], saved_reference_llms)
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
    reference_llm = choose_reference_llm(reference_llms, config, config_path)
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


def choose_reference_llm(
    reference_llms: list[ModelCandidate],
    config: dict | None = None,
    config_path: Path | None = None,
) -> ModelCandidate | None:
    print()
    print("LLM-corrected reference options:")
    print("[1] Use detected local GGUF reference LLM" + ("" if reference_llms else " (none detected yet)"))
    print("[2] Paste a GGUF file path or folder and save it for next run")
    print("[3] Use ChatGPT, Claude, or another external LLM manually")
    print("[4] Skip LLM reference for now")
    while True:
        choice = input("Reference option> ").strip().lower()
        if choice in {"", "4", "s", "skip", "n", "no"}:
            return None
        if choice in {"3", "manual", "external", "chatgpt", "claude"}:
            print_external_llm_guide()
            return None
        if choice in {"2", "path", "paste", "import", "manual import"}:
            if config is None or config_path is None:
                print("Custom LLM paths require config.json to be writable.")
                continue
            raw_path = input("GGUF file or folder path> ").strip()
            if not raw_path:
                continue
            try:
                new_candidates = save_custom_reference_path(config_path, config, raw_path)
            except (OSError, ValueError) as exc:
                print(f"Could not use that path: {exc}")
                continue
            reference_llms = merge_reference_llms(reference_llms, new_candidates)
            print(f"Saved path. Found {len(new_candidates)} GGUF reference LLM candidate(s).")
            return choose_reference_llm(reference_llms, config, config_path)
        if choice in {"1", "local", "detected", "gguf", "y", "yes"}:
            if not reference_llms:
                print("No GGUF reference LLMs were detected. Choose option 2 to paste a path, option 3 for external LLM instructions, or option 4 to skip.")
                continue
            print()
            print("Detected GGUF reference/correction LLMs:")
            for index, candidate in enumerate(reference_llms, 1):
                print(f"[{index}] {candidate.display_name} | {candidate.precision} | {candidate.path}")
            while True:
                raw = input("Reference LLM> ").strip()
                indexes = parse_selection(raw, len(reference_llms))
                if len(indexes) == 1:
                    return reference_llms[indexes[0] - 1]
                if raw.lower() in {"", "s", "skip"}:
                    return None
                print("Choose one reference LLM number, or press Enter to skip.")
        print("Choose 1, 2, 3, or 4.")
