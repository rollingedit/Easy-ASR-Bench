from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from verify_github_release import sha256
except ModuleNotFoundError:
    from scripts.verify_github_release import sha256


def build_manifest(tag: str, transcript: Path, assets: list[Path]) -> dict:
    return {
        "schema": "easy_asr_bench.release_verification_manifest.v1",
        "tag": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "self_hash_policy": "transcript_hash_excluded_from_transcript_recorded_here",
        "transcript": {
            "name": transcript.name,
            "sha256": sha256(transcript),
        },
        "assets": {
            asset.name: sha256(asset)
            for asset in sorted(assets, key=lambda item: item.name.lower())
            if asset.is_file() and asset.resolve() != transcript.resolve()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--asset", action="append", default=[], type=Path)
    parser.add_argument("--assets-dir", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assets = list(args.asset)
    for directory in args.assets_dir:
        assets.extend(path for path in directory.iterdir() if path.is_file())
    manifest = build_manifest(args.tag, args.transcript, assets)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
