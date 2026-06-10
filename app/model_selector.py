from __future__ import annotations

import re
from pathlib import Path

from .adapters.base import ModelCandidate
from .config import save_config
from .console_style import key, prompt_label
from .hf_model_downloader import download_hf_model_interactive
from .interactive_menu import MenuAction, choose_many, choose_one
from .llm_reference import merge_reference_llms, print_external_llm_guide, save_custom_reference_path, scan_custom_reference_llms
from .model_scanner import scan_models
from .model_status import candidate_reason, model_status_label


LAST_RUN_SELECTION_SCHEMA = "easy_asr_bench.last_run_selection.v1"


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
    if not candidates:
        return []
    selected: list[int] = []
    by_family: dict[str, list[tuple[int, ModelCandidate]]] = {}
    for index, candidate in enumerate(candidates, 1):
        family = (candidate.family_name or candidate.adapter_name or candidate.backend).lower()
        by_family.setdefault(family, []).append((index, candidate))
    precision_rank = {
        "int8": 0,
        "q8": 0,
        "fp16w": 1,
        "fp16": 1,
        "float16": 1,
        "f16": 1,
        "fp32": 2,
        "float32": 2,
        "f32": 2,
        "unknown": 3,
    }
    backend_rank = {"faster-whisper": 0, "whisper.cpp": 1, "transformers": 2, "onnxruntime": 3, "openai-whisper": 4}
    for group in by_family.values():
        ranked = sorted(
            group,
            key=lambda item: (
                precision_rank.get(item[1].precision, precision_rank.get(item[1].quantization_label, 5)),
                backend_rank.get(item[1].backend, 5),
                item[1].display_name.lower(),
            ),
        )
        selected.append(ranked[0][0])
    return sorted(selected[:4])


def resolve_last_run_selection(
    candidates: list[ModelCandidate],
    unsupported: list[ModelCandidate],
    config: dict | None,
) -> tuple[list[ModelCandidate], ModelCandidate | None, list[str]]:
    if config is None:
        return [], None, ["config unavailable"]
    state = config.get("last_run_selection")
    if not isinstance(state, dict) or state.get("schema") != LAST_RUN_SELECTION_SCHEMA:
        return [], None, ["no saved last-run selection"]
    asr_candidates = [candidate for candidate in candidates if candidate.category == "asr"]
    saved_reference_llms = scan_custom_reference_llms(config)
    reference_llms = merge_reference_llms([candidate for candidate in unsupported if candidate.category == "reference_llm"], saved_reference_llms)
    by_id = {candidate.candidate_id: candidate for candidate in asr_candidates}
    reference_by_id = {candidate.candidate_id: candidate for candidate in reference_llms}
    saved_ids = [str(item) for item in state.get("candidate_ids", []) if item]
    if not saved_ids:
        return [], None, ["saved last-run selection has no ASR model ids"]
    missing_ids = [candidate_id for candidate_id in saved_ids if candidate_id not in by_id]
    if missing_ids:
        return [], None, ["saved ASR model id not found: " + ", ".join(missing_ids)]
    reference_llm = None
    reference_id = str(state.get("reference_llm_candidate_id") or "")
    if reference_id:
        reference_llm = reference_by_id.get(reference_id)
        if reference_llm is None:
            return [], None, [f"saved reference LLM id not found: {reference_id}"]
    return [by_id[candidate_id] for candidate_id in saved_ids], reference_llm, []


def save_last_run_selection(
    config: dict | None,
    config_path: Path | None,
    selected: list[ModelCandidate],
    reference_llm: ModelCandidate | None,
) -> None:
    if config is None or config_path is None or not selected:
        return
    config["last_run_selection"] = {
        "schema": LAST_RUN_SELECTION_SCHEMA,
        "candidate_ids": [candidate.candidate_id for candidate in selected],
        "precision_buckets": sorted({candidate.quantization_label for candidate in selected}),
        "reference_llm_candidate_id": reference_llm.candidate_id if reference_llm else "",
    }
    save_config(config_path, config)


def choose_candidates(
    candidates: list[ModelCandidate],
    unsupported: list[ModelCandidate],
    config: dict | None = None,
    config_path: Path | None = None,
    models_root: Path | None = None,
) -> tuple[list[ModelCandidate], ModelCandidate | None]:
    while True:
        selected, reference_llm, should_rescan = _choose_candidates_once(candidates, unsupported, config, config_path, models_root)
        if not should_rescan:
            return selected, reference_llm
        if models_root is None:
            return [], None
        candidates, unsupported = scan_models(models_root)


def _choose_candidates_once(
    candidates: list[ModelCandidate],
    unsupported: list[ModelCandidate],
    config: dict | None = None,
    config_path: Path | None = None,
    models_root: Path | None = None,
) -> tuple[list[ModelCandidate], ModelCandidate | None, bool]:
    probe_candidates = [candidate for candidate in unsupported if candidate.category == "asr_probe_required"]
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
            f"[{key(str(index))}] {candidate.display_name:<30} backend: {candidate.backend:<12} "
            f"precision: {candidate.precision:<8} bucket: {candidate.quantization_label}"
        )
        print(f"    Path: {candidate.path}")
    if unsupported:
        if reference_llms:
            print()
            print("Detected reference/correction LLMs:")
            print()
            for index, candidate in enumerate(reference_llms, 1):
                print(f"[{key('L' + str(index))}] {candidate.display_name}    {candidate.backend}    {candidate.precision}    {candidate.quantization_label}")
                print(f"     Use: optional LLM-corrected reference generation, not direct transcription.")
        print()
        print("Detected non-runnable model packages:")
        print()
        for index, candidate in enumerate(unsupported, 1):
            print(f"[{key('U' + str(index))}] {candidate.display_name}    {model_status_label(candidate)}")
            print(f"     Reason: {candidate_reason(candidate)}")
    if models_root is not None:
        print()
        print(f"[{key('D')}] Paste a Hugging Face model link/repo id to download a model package")
    if not candidates:
        if models_root is None:
            return [], None, False
        menu_choice = choose_one(
            "No runnable ASR models were found.",
            ["Probe a complete unknown ASR folder", "Download a Hugging Face model package", "Stop for now"] if probe_candidates else ["Download a Hugging Face model package", "Stop for now"],
        )
        if probe_candidates and menu_choice == 0:
            selected = choose_probe_candidates(probe_candidates)
            reference_llm = choose_reference_llm(reference_llms, config, config_path)
            return selected, reference_llm, False
        if (probe_candidates and menu_choice == 1) or (not probe_candidates and menu_choice == 0):
            if download_hf_model_interactive(models_root) is not None:
                return [], None, True
        elif (probe_candidates and menu_choice == 2) or (not probe_candidates and menu_choice == 1):
            return [], None, False
        while True:
            raw = input(prompt_label("Models> ")).strip().lower()
            if raw in {"p", "probe"} and probe_candidates:
                selected = choose_probe_candidates(probe_candidates)
                reference_llm = choose_reference_llm(reference_llms, config, config_path)
                return selected, reference_llm, False
            if raw in {"d", "download", "hf", "huggingface"}:
                if download_hf_model_interactive(models_root) is not None:
                    return [], None, True
                continue
            if raw in {"", "q", "quit", "exit"}:
                return [], None, False
            probe_text = f", {key('P')} to probe a complete unknown ASR folder" if probe_candidates else ""
            print(f"No runnable models yet. Choose {key('D')} to download from Hugging Face{probe_text}, or press {key('Enter')} to stop.")
    print()
    saved_selected, saved_reference_llm, saved_errors = resolve_last_run_selection(candidates, unsupported + reference_llms, config)
    if saved_selected:
        print(
            "Saved last-run selection: "
            + ", ".join(candidate.display_name for candidate in saved_selected)
            + (f" + reference LLM {saved_reference_llm.display_name}" if saved_reference_llm else "")
        )
    if len(candidates) > 9:
        probe_text = f" Use {key('P')} to probe complete unknown ASR folders." if probe_candidates else ""
        last_text = f" Press {key('Enter')} to reuse last run." if saved_selected else ""
        print(f"Choose models with spaces, commas, or ranges: {key('1 2 10')}, {key('1,2,10')}, {key('1-4')}. Compact digits are disabled for 10+ models.{last_text} Use {key('D')} to download from Hugging Face.{probe_text}")
    else:
        probe_text = f", {key('P')} to probe complete unknown ASR folders" if probe_candidates else ""
        last_text = f", blank {key('Enter')} for last run" if saved_selected else ""
        print(f"Choose models: numbers like {key('1 2 4')}, {key('1,2,4')}, {key('1-4')}, {key('1234')}, {key('A')} for all, {key('R')} for recommended{last_text}, {key('D')} to download from Hugging Face{probe_text}.")
    menu_result = choose_many(
        "Choose ASR models",
        [f"{candidate.display_name} | {candidate.backend} | {candidate.precision} | {candidate.quantization_label}" for candidate in candidates],
        actions=[
            MenuAction("A", "select all"),
            MenuAction("R", "recommended"),
            *([MenuAction("L", "last run")] if saved_selected else []),
            *([MenuAction("P", "probe complete unknown ASR folder")] if probe_candidates else []),
            *([MenuAction("D", "download from Hugging Face")] if models_root is not None else []),
        ],
    )
    if menu_result == "d" and models_root is not None:
        if download_hf_model_interactive(models_root) is not None:
            return [], None, True
    elif menu_result == "a":
        indexes = list(range(1, len(candidates) + 1))
        selected = choose_precision_buckets([candidates[index - 1] for index in indexes])
        reference_llm = choose_reference_llm(reference_llms, config, config_path)
        save_last_run_selection(config, config_path, selected, reference_llm)
        return selected, reference_llm, False
    elif menu_result == "r":
        indexes = recommended_candidates(candidates)
        selected = choose_precision_buckets([candidates[index - 1] for index in indexes])
        reference_llm = choose_reference_llm(reference_llms, config, config_path)
        save_last_run_selection(config, config_path, selected, reference_llm)
        return selected, reference_llm, False
    elif menu_result == "l" and saved_selected:
        return saved_selected, saved_reference_llm, False
    elif menu_result == "p" and probe_candidates:
        selected = choose_probe_candidates(probe_candidates)
        reference_llm = choose_reference_llm(reference_llms, config, config_path)
        save_last_run_selection(config, config_path, selected, reference_llm)
        return selected, reference_llm, False
    elif isinstance(menu_result, list):
        selected = choose_precision_buckets([candidates[index] for index in menu_result])
        reference_llm = choose_reference_llm(reference_llms, config, config_path)
        save_last_run_selection(config, config_path, selected, reference_llm)
        return selected, reference_llm, False
    while True:
        raw = input(prompt_label("Models> ")).strip()
        if raw.lower() in {"d", "download", "hf", "huggingface"} and models_root is not None:
            if download_hf_model_interactive(models_root) is not None:
                return [], None, True
            continue
        if raw.lower() in {"p", "probe"} and probe_candidates:
            selected = choose_probe_candidates(probe_candidates)
            reference_llm = choose_reference_llm(reference_llms, config, config_path)
            return selected, reference_llm, False
        if raw.lower() in {"r", "recommended"}:
            indexes = recommended_candidates(candidates)
        elif not raw and saved_selected:
            return saved_selected, saved_reference_llm, False
        else:
            indexes = parse_selection(raw, len(candidates))
        if indexes:
            break
        print("No valid runnable models selected.")
    selected = choose_precision_buckets([candidates[index - 1] for index in indexes])
    reference_llm = choose_reference_llm(reference_llms, config, config_path)
    save_last_run_selection(config, config_path, selected, reference_llm)
    return selected, reference_llm, False


def choose_probe_candidates(candidates: list[ModelCandidate]) -> list[ModelCandidate]:
    print()
    print("Complete unknown ASR folders available for runtime probe:")
    for index, candidate in enumerate(candidates, 1):
        print(f"[{key(str(index))}] {candidate.display_name} | {candidate.backend} | {candidate.path}")
    menu_result = choose_many(
        "Choose ASR folders to probe",
        [f"{candidate.display_name} | {candidate.backend} | {candidate.path}" for candidate in candidates],
    )
    if isinstance(menu_result, list):
        return [candidates[index] for index in menu_result]
    while True:
        raw = input(prompt_label("Probe> ")).strip()
        indexes = parse_selection(raw, len(candidates))
        if indexes:
            return [candidates[index - 1] for index in indexes]
        if raw.lower() in {"", "q", "quit", "exit"}:
            return []
        print("Choose one or more probe candidate numbers, or press Enter to cancel.")


def choose_precision_buckets(candidates: list[ModelCandidate]) -> list[ModelCandidate]:
    buckets = sorted({candidate.quantization_label for candidate in candidates})
    print()
    print("Available precision buckets in selected models:")
    print()
    for index, bucket in enumerate(buckets, 1):
        print(f"[{key(str(index))}] {bucket}")
    print(f"[{key('A')}] All available precisions")
    menu_result = choose_many("Choose precision buckets", buckets, actions=[MenuAction("A", "all available precisions")])
    if menu_result == "a":
        return candidates
    if isinstance(menu_result, list):
        chosen = {buckets[index] for index in menu_result}
        return [candidate for candidate in candidates if candidate.quantization_label in chosen]
    while True:
        raw = input(prompt_label("Precisions> ")).strip()
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
    print(f"[{key('1')}] Use detected local GGUF reference LLM" + ("" if reference_llms else " (none detected yet)"))
    print(f"[{key('2')}] Paste a GGUF file path or folder and save it for next run")
    print(f"[{key('3')}] Use ChatGPT, Claude, or another external LLM manually")
    print(f"[{key('4')}] Skip LLM reference for now")
    menu_result = choose_one(
        "Choose LLM-corrected reference option",
        [
            "Use detected local GGUF reference LLM" + ("" if reference_llms else " (none detected yet)"),
            "Paste a GGUF file path or folder and save it for next run",
            "Use ChatGPT, Claude, or another external LLM manually",
            "Skip LLM reference for now",
        ],
    )
    if menu_result == 0:
        choice = "1"
    elif menu_result == 1:
        choice = "2"
    elif menu_result == 2:
        choice = "3"
    elif menu_result == 3:
        return None
    else:
        choice = None
    while True:
        if choice is None:
            choice = input(prompt_label("Reference option> ")).strip().lower()
        if choice in {"", "4", "s", "skip", "n", "no"}:
            return None
        if choice in {"3", "manual", "external", "chatgpt", "claude"}:
            print_external_llm_guide()
            return None
        if choice in {"2", "path", "paste", "import", "manual import"}:
            if config is None or config_path is None:
                print("Custom LLM paths require config.json to be writable.")
                choice = None
                continue
            raw_path = input(prompt_label("GGUF file or folder path> ")).strip()
            if not raw_path:
                choice = None
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
                choice = None
                continue
            print()
            print("Detected GGUF reference/correction LLMs:")
            for index, candidate in enumerate(reference_llms, 1):
                print(f"[{key(str(index))}] {candidate.display_name} | {candidate.precision} | {candidate.path}")
            menu_llm = choose_one(
                "Choose local GGUF reference LLM",
                [f"{candidate.display_name} | {candidate.precision} | {candidate.path}" for candidate in reference_llms],
            )
            if isinstance(menu_llm, int):
                return reference_llms[menu_llm]
            while True:
                raw = input(prompt_label("Reference LLM> ")).strip()
                indexes = parse_selection(raw, len(reference_llms))
                if len(indexes) == 1:
                    return reference_llms[indexes[0] - 1]
                if raw.lower() in {"", "s", "skip"}:
                    return None
                print("Choose one reference LLM number, or press Enter to skip.")
        print(f"Choose {key('1')}, {key('2')}, {key('3')}, or {key('4')}.")
        choice = None
