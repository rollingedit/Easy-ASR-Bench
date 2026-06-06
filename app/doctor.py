from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .dependency_manager import dependency_status


def run_doctor(config_path: Path) -> int:
    config = load_config(config_path)
    folders = config["folders"]
    for folder in folders.values():
        Path(folder).mkdir(parents=True, exist_ok=True)
    status = dependency_status()
    print("Easy ASR Bench Doctor")
    print()
    for group, data in status.items():
        mark = "OK" if data["available"] else "MISSING"
        print(f"{mark:8} {group}")
        if data["missing"]:
            print("         missing: " + ", ".join(data["missing"]))
    print()
    print("Folders checked:")
    for key, folder in folders.items():
        print(f"  {key}: {folder}")
    return 0 if status["core"]["available"] else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    raise SystemExit(run_doctor(Path(args.config)))


if __name__ == "__main__":
    main()
