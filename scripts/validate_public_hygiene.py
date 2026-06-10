from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sensitive_terms() -> list[str]:
    return [
        "40" + "90",
        "agents" + ".md",
        "private" + " hand" + "off",
        "private" + "_hand" + "off",
        "LOCAL" + "_RELEASE_QA",
        "PUBLIC" + "_COMMIT",
        "Windows Pro " + "RTX",
        "rollingedit/Easy-ASR-Bench-" + "private-hand" + "off",
        "hand" + "off",
        "this " + "laptop",
        "my " + "laptop",
        "non-NVIDIA " + "laptop",
        "AMD/DirectML " + "laptop",
        "diffusion-audio-" + "restoration",
        "C:" + "\\Users\\" + "PC",
        "C:" + "\\Users\\" + "computer",
    ]


def _sensitive_patterns() -> list[re.Pattern[str]]:
    patterns = [re.compile(re.escape(term), re.IGNORECASE) for term in _sensitive_terms()]
    patterns.extend(
        [
            re.compile(r"NVIDIA\s+GeForce\s+RTX\s+\d{4}(?:\s+\w+)?", re.IGNORECASE),
            re.compile(r"Intel\(R\)\s+UHD\s+Graphics\s+\d+", re.IGNORECASE),
            re.compile(r"\b\d{1,2}th\s+Gen\s+Intel\(R\)\s+Core\(TM\)\s+i\d+-\d+\w*\b", re.IGNORECASE),
        ]
    )
    return patterns


def tracked_files(root: Path = ROOT) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    return [root / line for line in completed.stdout.splitlines() if line.strip()]


def _line_matches(path: Path, text: str) -> list[str]:
    findings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern in _sensitive_patterns():
            if pattern.search(line):
                findings.append(f"{path}:{lineno}: {pattern.pattern}")
                break
    return findings


def scan_paths(paths: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            findings.append(f"{path}: missing file")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        findings.extend(_line_matches(path, text))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan public release files for local/private validation details.")
    parser.add_argument(
        "--tracked",
        action="store_true",
        help="Scan all tracked text files in the repository.",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        type=Path,
        help="Additional generated public file to scan, such as release-smoke-vX.Y.Z.json.",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if args.tracked or not args.path:
        paths.extend(tracked_files())
    paths.extend(args.path)
    findings = scan_paths(paths)
    if findings:
        print("public hygiene validation failed:")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    print("public hygiene validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
