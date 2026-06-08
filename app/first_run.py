from __future__ import annotations

import os
from pathlib import Path

from .console_style import key, prompt_label
from .hf_model_downloader import download_hf_model_interactive, download_recommended_baseline
from .interactive_menu import MenuAction, choose_one
from .model_scanner import scan_models
from .model_status import model_status_label
from .repair_plan import build_repair_plan
from .version import RELEASE_CHANNEL, RELEASE_COMMIT, TAG


def run_first_run_wizard(config: dict, *, input_func=input, print_func=print, initial_action: str | None = None) -> bool:
    models_root = Path(config["folders"]["models"])
    input_root = Path(config["folders"]["input"])
    runnable, unsupported = scan_models(models_root)
    runnable_asr = [candidate for candidate in runnable if candidate.category == "asr"]
    incomplete = [candidate for candidate in unsupported if model_status_label(candidate) == "Recognized incomplete"]
    reference_llms = [candidate for candidate in unsupported if candidate.category == "reference_llm"]

    print_func("Easy ASR Bench is ready.")
    print_func()
    print_func("Nothing will be uploaded. Models and media stay local.")
    print_func("Network use: GitHub/PyPI during setup, Hugging Face only if you choose a model download.")
    print_func()
    if initial_action == "paste_hf":
        return download_hf_model_interactive(models_root, input_func=input_func, print_func=print_func) is not None
    if initial_action == "recommended_baseline":
        return download_recommended_baseline(models_root, input_func=input_func, print_func=print_func) is not None
    if runnable_asr:
        print_func(f"Runnable ASR models found: {len(runnable_asr)}")
        action = _choose_action(
            "Next step",
            [
                "Run Easy ASR Bench now",
                "Paste a Hugging Face model link to download another model",
                "Open Input folder",
                "Open Models folder",
                "Quit",
            ],
            {"r": 0, "p": 1, "i": 2, "m": 3, "q": 4},
            input_func,
            print_func,
        )
        if action == 0:
            return True
        if action == 1:
            return download_hf_model_interactive(models_root, input_func=input_func, print_func=print_func) is not None
        if action == 2:
            _open_folder(input_root)
        elif action == 3:
            _open_folder(models_root)
        return False

    print_func("No runnable ASR model is installed yet.")
    print_func()
    print_func("Detected files:")
    print_func(f"  - {len(incomplete)} incomplete model folder(s)")
    print_func(f"  - {len(reference_llms)} reference/correction LLM(s)")
    print_func("  - 0 runnable ASR models")
    print_func()
    print_func("Recommended CPU baseline:")
    print_func("  Downloads: Systran/faster-whisper-tiny.en")
    print_func("  Installs when selected: faster-whisper / CTranslate2 runtime")
    print_func("  Runs on CPU by default; CUDA is optional and only used if available.")
    print_func()
    action = _choose_action(
        "Choose one",
        [
            "Download recommended CPU baseline",
            "Paste a Hugging Face model link",
            "Open Models folder and rescan later",
            "Open Input folder",
            "Quit",
        ],
        {"d": 0, "p": 1, "m": 2, "i": 3, "q": 4},
        input_func,
        print_func,
    )
    if action == 0:
        return download_recommended_baseline(models_root, input_func=input_func, print_func=print_func) is not None
    if action == 1:
        return download_hf_model_interactive(models_root, input_func=input_func, print_func=print_func) is not None
    if action == 2:
        _open_folder(models_root)
    elif action == 3:
        _open_folder(input_root)
    return False


def build_first_run_smoke_report(config: dict) -> dict:
    models_root = Path(config["folders"]["models"])
    input_root = Path(config["folders"]["input"])
    models_root.mkdir(parents=True, exist_ok=True)
    input_root.mkdir(parents=True, exist_ok=True)
    runnable, unsupported = scan_models(models_root)
    runnable_asr = [candidate for candidate in runnable if candidate.category == "asr"]
    incomplete = [candidate for candidate in unsupported if model_status_label(candidate) == "Recognized incomplete"]
    reference_llms = [candidate for candidate in unsupported if candidate.category == "reference_llm"]
    repair_plan = build_repair_plan(config)
    return {
        "schema": "easy_asr_bench.first_run_smoke.v1",
        "version": TAG,
        "release_channel": RELEASE_CHANNEL,
        "release_commit": RELEASE_COMMIT,
        "models_root": str(models_root),
        "input_root": str(input_root),
        "runnable_asr_count": len(runnable_asr),
        "incomplete_model_count": len(incomplete),
        "reference_llm_count": len(reference_llms),
        "network_used": False,
        "repair_plan_schema": repair_plan["schema"],
        "repair_plan_summary": repair_plan["summary"],
        "repair_command": "setup.bat --doctor --repair-all-safe",
        "doctor_command": "setup.bat --doctor --repair-plan",
        "real_smoke_command": "setup.bat --doctor --validate-real-smoke",
        "available_actions": ["run_now", "download_recommended_baseline", "paste_hugging_face_link", "open_models_folder", "open_input_folder", "quit"],
        "recommended_next_action": "run_now" if runnable_asr else "download_recommended_baseline",
        "dead_end": False,
    }


def _choose_action(title: str, options: list[str], typed: dict[str, int], input_func, print_func) -> int | None:
    actions = [MenuAction(letter.upper(), label.split(" ", 1)[0].lower()) for letter, label in zip(typed, options)]
    menu_result = choose_one(title, options, actions=actions)
    if isinstance(menu_result, int):
        return menu_result
    if isinstance(menu_result, str) and menu_result.lower() in typed:
        return typed[menu_result.lower()]
    print_func(title + ":")
    for letter, index in typed.items():
        print_func(f"  [{key(letter.upper())}] {options[index]}")
    while True:
        raw = input_func(prompt_label("First run> ")).strip().lower()
        if raw in typed:
            return typed[raw]
        if raw in {"", "q", "quit", "exit"}:
            return typed.get("q")
        print_func("Choose one of: " + ", ".join(key(letter.upper()) for letter in typed))


def _open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        print(path)
