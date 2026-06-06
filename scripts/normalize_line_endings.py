from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CRLF_SUFFIXES = {".bat", ".cmd", ".ps1"}
LF_SUFFIXES = {".py", ".json", ".md", ".toml", ".ini", ".yml", ".yaml", ".html", ".css", ".js", ".txt"}
LF_NAMES = {".gitattributes", ".gitignore", ".editorconfig", "license"}


def git_files() -> list[Path]:
    completed = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=True)
    return [ROOT / line.strip() for line in completed.stdout.splitlines() if line.strip()]


def normalize(path: Path) -> bool:
    data = path.read_bytes()
    if b"\0" in data:
        return False
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in CRLF_SUFFIXES:
        newline = b"\r\n"
    elif suffix in LF_SUFFIXES or name in LF_NAMES:
        newline = b"\n"
    else:
        return False
    text = data.decode("utf-8")
    updated = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8").replace(b"\n", newline)
    if updated != data:
        path.write_bytes(updated)
        return True
    return False


def main() -> int:
    changed = [path.relative_to(ROOT).as_posix() for path in git_files() if path.is_file() and normalize(path)]
    if changed:
        print("normalized line endings:")
        for rel in changed:
            print(f"  {rel}")
    else:
        print("line endings already normalized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
