from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .reference_schema import validate_llm_reference
except ImportError:
    from reference_schema import validate_llm_reference


def merge_parts(parts: list[dict], results: dict | None = None) -> dict:
    merged = {
        "schema": "easy_asr_bench.llm_reference.v1",
        "source_sha256": "",
        "reference_type": "llm_corrected_reference",
        "segments": [],
        "global_notes": [],
    }
    for index, data in enumerate(parts, 1):
        if data.get("schema") != "easy_asr_bench.llm_reference.v1":
            raise ValueError(f"Part {index} has invalid schema.")
        if data.get("reference_type") != "llm_corrected_reference":
            raise ValueError(f"Part {index} has invalid reference_type.")
        if data.get("source_sha256"):
            if merged["source_sha256"] and merged["source_sha256"] != data["source_sha256"]:
                raise ValueError(f"Part {index} source_sha256 does not match earlier parts.")
            merged["source_sha256"] = data["source_sha256"]
        merged["segments"].extend(data.get("segments", []))
        merged["global_notes"].extend(data.get("global_notes", []))
    if results is not None:
        errors = validate_llm_reference(merged, results)
        if errors:
            raise ValueError("Merged reference is invalid: " + "; ".join(errors))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="+")
    parser.add_argument("--output", default="merged_reference.json")
    parser.add_argument("--results", default="")
    args = parser.parse_args()
    parts = [json.loads(Path(item).read_text(encoding="utf-8")) for item in args.parts]
    results = json.loads(Path(args.results).read_text(encoding="utf-8")) if args.results else None
    merged = merge_parts(parts, results)
    Path(args.output).write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
