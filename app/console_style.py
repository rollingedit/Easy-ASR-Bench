from __future__ import annotations

import os
import sys


KEY_COLOR = "\033[96;1m"
PROMPT_COLOR = "\033[93;1m"
RESET = "\033[0m"


def color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(os.environ.get("FORCE_COLOR")) or sys.stdout.isatty()


def colored(text: str, color: str = KEY_COLOR) -> str:
    if not color_enabled():
        return text
    return f"{color}{text}{RESET}"


def key(text: str) -> str:
    return colored(text, KEY_COLOR)


def prompt_label(text: str) -> str:
    return colored(text, PROMPT_COLOR)
