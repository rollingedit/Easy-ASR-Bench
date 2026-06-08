from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa.runtime_matrix.registry import ROWS


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Easy ASR Bench runtime/release matrix row.")
    parser.add_argument("--row", required=True, choices=sorted(ROWS))
    parser.add_argument("--workdir", type=Path, default=ROOT / "Temp" / "runtime_matrix")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true")
    args = parser.parse_args()

    definition = ROWS[args.row]
    module = importlib.import_module(definition.module)
    evidence_dir = args.workdir / args.row
    result = module.run(args.row, evidence_dir, args.install_deps, args.allow_downloads)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in {"pass", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

