from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Sequence

from .console_style import key


@dataclass(frozen=True)
class MenuAction:
    key_name: str
    label: str


def menu_supported() -> bool:
    return os.name == "nt" and sys.stdin.isatty() and sys.stdout.isatty()


def _read_key() -> str:
    import msvcrt

    char = msvcrt.getwch()
    if char in {"\x00", "\xe0"}:
        second = msvcrt.getwch()
        return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(second, "")
    if char == "\r":
        return "enter"
    if char == " ":
        return "space"
    if char == "\x1b":
        return "escape"
    return char.lower()


def _clear() -> None:
    print("\033[2J\033[H", end="")


def choose_one(title: str, options: Sequence[str], *, actions: Sequence[MenuAction] = ()) -> int | str | None:
    if not menu_supported() or not options:
        return None
    selected = 0
    while True:
        _clear()
        print(title)
        print(f"{key('Up/Down')} move  {key('Enter')} choose" + (f"  {key('Esc')} cancel" if actions else ""))
        print()
        for index, label in enumerate(options):
            prefix = key(">") if index == selected else " "
            print(f"{prefix} {label}")
        if actions:
            print()
            print("Other actions:")
            for action in actions:
                print(f"  {key(action.key_name)} {action.label}")
        pressed = _read_key()
        if pressed == "up":
            selected = (selected - 1) % len(options)
        elif pressed == "down":
            selected = (selected + 1) % len(options)
        elif pressed == "enter":
            return selected
        elif pressed == "escape":
            return None
        for action in actions:
            if pressed == action.key_name.lower():
                return action.key_name.lower()


def choose_many(title: str, options: Sequence[str], *, actions: Sequence[MenuAction] = ()) -> list[int] | str | None:
    if not menu_supported() or not options:
        return None
    selected = 0
    checked: set[int] = set()
    while True:
        _clear()
        print(title)
        print(f"{key('Up/Down')} move  {key('Space')} toggle  {key('Enter')} confirm")
        print()
        for index, label in enumerate(options):
            cursor = key(">") if index == selected else " "
            mark = key("x") if index in checked else " "
            print(f"{cursor} [{mark}] {label}")
        if actions:
            print()
            print("Other actions:")
            for action in actions:
                print(f"  {key(action.key_name)} {action.label}")
        pressed = _read_key()
        if pressed == "up":
            selected = (selected - 1) % len(options)
        elif pressed == "down":
            selected = (selected + 1) % len(options)
        elif pressed == "space":
            if selected in checked:
                checked.remove(selected)
            else:
                checked.add(selected)
        elif pressed == "enter":
            return sorted(checked) if checked else [selected]
        elif pressed == "escape":
            return None
        for action in actions:
            if pressed == action.key_name.lower():
                return action.key_name.lower()
