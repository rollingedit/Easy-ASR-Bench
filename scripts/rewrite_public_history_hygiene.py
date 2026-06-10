from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _byte_replacements() -> list[tuple[re.Pattern[bytes], bytes]]:
    exact = [
        ("40" + "90", "CUDA dGPU"),
        ("agents" + ".md", "local instructions"),
        ("private" + " hand" + "off", "local-only transfer"),
        ("private" + "_hand" + "off", "local_transfer"),
        ("LOCAL" + "_RELEASE_QA", "LOCAL_QA"),
        ("PUBLIC" + "_COMMIT", "PUBLIC_REVISION"),
        ("Windows Pro " + "RTX", "Windows CUDA validation host"),
        ("rollingedit/Easy-ASR-Bench-" + "private-hand" + "off", "local-transfer-repo"),
        ("hand" + "off", "local transfer"),
        ("this " + "laptop", "this validation host"),
        ("my " + "laptop", "my validation host"),
        ("non-NVIDIA " + "laptop", "non-NVIDIA validation host"),
        ("AMD/DirectML " + "laptop", "DirectML validation host"),
        ("diffusion-audio-" + "restoration", "unrelated-workspace"),
        ("C:" + "\\Users\\" + "PC", "%USERPROFILE%"),
        ("C:" + "\\Users\\" + "computer", "%USERPROFILE%"),
    ]
    replacements = [(re.compile(re.escape(old.encode("utf-8")), re.IGNORECASE), new.encode("utf-8")) for old, new in exact]
    replacements.extend(
        [
            (re.compile(rb"NVIDIA\s+GeForce\s+RTX\s+\d{4}(?:\s+\w+)?", re.IGNORECASE), b"NVIDIA CUDA GPU"),
            (re.compile(rb"NVIDIA\s+RTX\s+\d{4}(?:\s+\w+)?", re.IGNORECASE), b"NVIDIA CUDA GPU"),
            (re.compile(rb"Intel\(R\)\s+UHD\s+Graphics\s+\d+", re.IGNORECASE), b"Intel integrated GPU"),
            (
                re.compile(rb"\b\d{1,2}th\s+Gen\s+Intel\(R\)\s+Core\(TM\)\s+i\d+-\d+\w*\b", re.IGNORECASE),
                b"Intel CPU",
            ),
        ]
    )
    return replacements


def sanitize_bytes(data: bytes) -> bytes:
    output = data
    for pattern, replacement in _byte_replacements():
        output = pattern.sub(replacement, output)
    return output


def sanitize_fast_export(stream: bytes) -> bytes:
    output = bytearray()
    index = 0
    marker = b"data "
    while index < len(stream):
        line_end = stream.find(b"\n", index)
        if line_end == -1:
            output.extend(stream[index:])
            break
        line = stream[index:line_end]
        output.extend(stream[index : line_end + 1])
        index = line_end + 1
        if not line.startswith(marker):
            continue
        size_text = line[len(marker) :].strip()
        if not size_text.isdigit():
            continue
        size = int(size_text)
        payload = stream[index : index + size]
        sanitized = sanitize_bytes(payload)
        del output[-(len(line) + 1) :]
        output.extend(f"data {len(sanitized)}\n".encode("ascii"))
        output.extend(sanitized)
        index += size
    return bytes(output)


def _run(command: list[str], cwd: Path, *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, input=input_bytes, capture_output=True, check=True)


def _remove_tree(path: Path) -> None:
    def on_error(function, name, _exc_info):
        os.chmod(name, stat.S_IWRITE)
        function(name)

    shutil.rmtree(path, onerror=on_error)


def rewrite_to_candidate_repo(source_ref: str, target_dir: Path, source_root: Path = ROOT, *, force: bool = False) -> None:
    target_dir = target_dir.resolve()
    if target_dir.exists() and any(target_dir.iterdir()):
        if not force:
            raise SystemExit(f"Target directory is not empty; pass --force to replace it: {target_dir}")
        _remove_tree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    _run(["git", "init"], target_dir)
    exported = _run(["git", "fast-export", "--signed-tags=strip", "--tag-of-filtered-object=rewrite", source_ref], source_root).stdout
    _run(["git", "fast-import", "--quiet"], target_dir, input_bytes=sanitize_fast_export(exported))
    refs = _run(["git", "for-each-ref", "--format=%(refname:short)", "refs/heads"], target_dir).stdout.decode("utf-8").splitlines()
    if "main" in refs:
        _run(["git", "checkout", "-f", "main"], target_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a sanitized candidate repo for public-history cleanup review.")
    parser.add_argument("--source-ref", default="main", help="Source ref to export and sanitize.")
    parser.add_argument("--target-dir", required=True, type=Path, help="Empty target directory for the sanitized candidate repo.")
    parser.add_argument("--force", action="store_true", help="Replace a non-empty target directory.")
    args = parser.parse_args()
    rewrite_to_candidate_repo(args.source_ref, args.target_dir, force=args.force)
    print(args.target_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
