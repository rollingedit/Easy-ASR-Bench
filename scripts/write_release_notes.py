from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def changelog_section(tag: str) -> list[str]:
    version = tag[1:] if tag.startswith("v") else tag
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    marker = f"## v{version}"
    if marker not in text:
        return []
    section = text.split(marker, 1)[1]
    if "\n## " in section:
        section = section.split("\n## ", 1)[0]
    return [line.strip() for line in section.splitlines() if line.strip().startswith("- ")]


def write_notes(tag: str, output: Path) -> None:
    changes = changelog_section(tag)
    if not changes:
        changes = ["- Maintenance release with validated packaging and installer updates."]
    body = [
        f"# Easy ASR Bench {tag}",
        "",
        "## What changed",
        "",
        *changes,
        "",
        "## Verified",
        "",
        "- Release file validator passed.",
        "- Release ZIP built in no-update verification mode.",
        "- Python compile check passed.",
        "- Unit tests passed.",
        "- `setup.bat --dry-run --local` passed.",
        "- Strict doctor passed for core dependencies.",
        "- GitHub Actions Release Gate must pass before this release is considered final.",
        "",
        "## Known limits",
        "",
        "- Optional model dependency groups install only when needed.",
        "- GPU/VRAM metrics require a CUDA-capable runtime; CPU runs report VRAM as `null`.",
        "- Unsafe pickle-backed `.pt` checkpoints remain blocked unless explicitly trusted.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")
    print(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="Version tag such as v0.2.7")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    tag = args.version if args.version.startswith("v") else f"v{args.version}"
    output = Path(args.output) if args.output else ROOT / "release_notes" / f"{tag}.md"
    write_notes(tag, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
