from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="+")
    parser.add_argument("--output", default="merged_reference.json")
    args = parser.parse_args()
    merged = {
        "schema": "easy_asr_bench.llm_reference.v1",
        "source_sha256": "",
        "reference_type": "llm_corrected_reference",
        "segments": [],
        "global_notes": [],
    }
    for item in args.parts:
        data = json.loads(Path(item).read_text(encoding="utf-8"))
        if data.get("source_sha256"):
            merged["source_sha256"] = data["source_sha256"]
        merged["segments"].extend(data.get("segments", []))
        merged["global_notes"].extend(data.get("global_notes", []))
    Path(args.output).write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
