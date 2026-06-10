from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from .config import load_config
from .version import TAG


LATEST_RELEASE_API = "https://api.github.com/repos/rollingedit/Easy-ASR-Bench/releases/latest"


def _version_tuple(tag: str) -> tuple[int, ...]:
    value = tag.strip().lstrip("vV")
    parts = []
    for item in value.split("."):
        match = re.match(r"(\d+)", item)
        if not match:
            break
        parts.append(int(match.group(1)))
    return tuple(parts)


def _latest_release_tag(timeout_seconds: float = 5.0) -> str:
    request = urllib.request.Request(
        LATEST_RELEASE_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "Easy-ASR-Bench-update-check"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))
    tag = data.get("tag_name")
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("GitHub latest-release response did not include tag_name")
    return tag.strip()


def check_for_updates(*, current_tag: str = TAG, timeout_seconds: float = 5.0, print_func=print) -> dict:
    try:
        latest = _latest_release_tag(timeout_seconds)
    except (OSError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        message = f"Update check unavailable: {exc}"
        print_func(message)
        return {"status": "unavailable", "current": current_tag, "latest": "", "message": message}
    current_version = _version_tuple(current_tag)
    latest_version = _version_tuple(latest)
    if latest_version and current_version and latest_version > current_version:
        message = f"Update available: {latest}. Download it from https://github.com/rollingedit/Easy-ASR-Bench/releases/latest"
        print_func(message)
        return {"status": "update_available", "current": current_tag, "latest": latest, "message": message}
    message = f"Easy ASR Bench is up to date ({current_tag})."
    print_func(message)
    return {"status": "current", "current": current_tag, "latest": latest, "message": message}


def check_for_updates_from_config(config: dict, *, context: str, print_func=print) -> dict | None:
    app_config = config.get("app", {})
    flag = "check_for_updates_on_setup" if context == "setup" else "check_for_updates_on_run"
    if not app_config.get(flag, False):
        return None
    return check_for_updates(print_func=print_func)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--context", choices=["setup", "run"], default="run")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    check_for_updates_from_config(config, context=args.context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
