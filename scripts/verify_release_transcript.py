from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from verify_github_release import sha256
except ModuleNotFoundError:
    from scripts.verify_github_release import sha256


ASSET_LINE = re.compile(r"^\s{2}(.+?)\s+(sha256:[0-9a-f]{64})$", re.IGNORECASE)


def transcript_hashes(text: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in text.splitlines():
        match = ASSET_LINE.match(line)
        if match:
            hashes[match.group(1)] = match.group(2).lower()
    return hashes


def verify_transcript(
    assets_dir: Path,
    checksums_path: Path,
    transcript_path: Path,
    strict: bool = False,
    detached_manifest_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    transcript = transcript_path.read_text(encoding="utf-8")
    if "Easy ASR Bench release verification transcript" not in transcript:
        errors.append("transcript header is missing")
    hashes = transcript_hashes(transcript)
    if transcript_path.name in hashes:
        errors.append("transcript must not include a self-hash; self hashes belong in a detached manifest")
    checksums = json.loads(checksums_path.read_text(encoding="utf-8")).get("files", {})
    for name, expected in checksums.items():
        asset = assets_dir / name
        if not asset.exists():
            errors.append(f"checksums.json names missing asset: {name}")
            continue
        actual = sha256(asset)
        if actual != expected:
            errors.append(f"checksum mismatch for {name}: expected {expected}, got {actual}")
        if name not in hashes:
            errors.append(f"transcript is missing asset hash for {name}")
        elif hashes[name] != actual:
            errors.append(f"transcript hash mismatch for {name}: expected {actual}, got {hashes[name]}")
    if strict:
        for name in hashes:
            if name not in checksums and name != transcript_path.name:
                asset = assets_dir / name
                if not asset.exists():
                    errors.append(f"transcript names missing asset: {name}")
                elif sha256(asset) != hashes[name]:
                    errors.append(f"transcript hash mismatch for {name}")
    if detached_manifest_path is not None:
        manifest = json.loads(detached_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "easy_asr_bench.release_verification_manifest.v1":
            errors.append("detached verification manifest has an unexpected schema")
        expected_transcript = manifest.get("transcript", {})
        if expected_transcript.get("name") != transcript_path.name:
            errors.append("detached verification manifest names the wrong transcript")
        elif expected_transcript.get("sha256") != sha256(transcript_path):
            errors.append("detached verification manifest transcript hash mismatch")
        for name, digest in manifest.get("assets", {}).items():
            asset = assets_dir / name
            if not asset.exists():
                errors.append(f"detached verification manifest names missing asset: {name}")
            elif sha256(asset) != digest:
                errors.append(f"detached verification manifest asset hash mismatch for {name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--checksums", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--detached-manifest", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    errors = verify_transcript(args.assets_dir, args.checksums, args.transcript, args.strict, args.detached_manifest)
    if errors:
        print("release transcript validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("release transcript validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
