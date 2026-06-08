from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".venv",
    "dist",
    ".pytest_cache",
    ".pytest_tmp",
    "__pycache__",
    "Logs",
    "Output",
    "Temp",
    "Cache",
    "Models",
    "Input",
}
SKIP_DIR_PREFIXES = (".pytest_tmp",)

REQUIRED_FILES = {
    "setup.bat",
    "Run.bat",
    "Drop_Audio_Or_Folders_Here.bat",
    "installer/install.ps1",
    "app/main.py",
    "app/model_scanner.py",
    "app/results_writer.py",
    "app/scoring.py",
    "app/hf_model_downloader.py",
    "scripts/validate_physical_files.py",
    "scripts/verify_github_release.py",
    ".github/workflows/release-gate.yml",
    ".github/workflows/publish-release.yml",
    "requirements/core.txt",
    "config.json",
}

REQUIRED_TEXT_MARKERS = {
    "setup.bat": ["APP_VERSION=", "--dry-run", "--verify-release", "--local"],
    "Run.bat": ["python -m app.main", "--doctor", "--first-run-smoke"],
    "Drop_Audio_Or_Folders_Here.bat": ["app.main"],
    "installer/install.ps1": ["Move-PreservedUserData", "Restore-MovedUserData", "Assert-StagingPhysicalFiles"],
    "app/main.py": ["def process_file_with_candidates", "def main()", "build_model_failure_error"],
    "app/model_scanner.py": ["def scan_models", "ModelCandidate", "indexed_safetensor_missing_files"],
    "app/results_writer.py": ["def build_results", "def write_all_reports", "runtime_rankings"],
    "app/scoring.py": ["def edit_distance", "def balanced_score", "def score_against_reference"],
    "app/hf_model_downloader.py": ["def download_hf_model_from_ref", "RECOMMENDED_BASELINE_REPO", "offer_missing_file_repair"],
    "scripts/validate_physical_files.py": ["REQUIRED_TEXT_MARKERS", "def validate_root"],
    "scripts/verify_github_release.py": ["def verify_release", "release-smoke", "checksums.json"],
    ".github/workflows/release-gate.yml": ["Validate release files", "Run unit tests"],
    ".github/workflows/publish-release.yml": ["Publish verified release", "gh release upload"],
    "requirements/core.txt": ["numpy", "huggingface_hub"],
    "config.json": ['"folders"', '"runtime"', '"chunking"'],
}

CRLF_SUFFIXES = {".bat", ".cmd", ".ps1"}
LF_SUFFIXES = {".py", ".json", ".md", ".toml", ".ini", ".yml", ".yaml", ".html", ".css", ".js", ".txt"}
LF_NAMES = {".gitattributes", ".gitignore", ".editorconfig", "license"}


def _is_skipped(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    return bool(SKIP_DIRS & set(rel.parts)) or any(part.startswith(SKIP_DIR_PREFIXES) for part in rel.parts)


def _raw(path: Path) -> bytes:
    return path.read_bytes()


def _physical_lines(path: Path) -> int:
    return len(_raw(path).splitlines())


def _assert_no_cr_only(path: Path) -> None:
    data = _raw(path)
    if b"\r" in data.replace(b"\r\n", b""):
        raise AssertionError(f"{path} contains CR-only line endings")


def _assert_line_ending(path: Path, expected: bytes) -> None:
    data = _raw(path)
    if not data:
        return
    _assert_no_cr_only(path)
    if expected == b"\r\n" and b"\n" in data.replace(b"\r\n", b""):
        raise AssertionError(f"{path} must use CRLF")
    if expected == b"\n" and b"\r\n" in data:
        raise AssertionError(f"{path} must use LF")


def _iter_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and not _is_skipped(path, root)]


def _validate_required_files(root: Path) -> None:
    for rel in sorted(REQUIRED_FILES):
        path = root / rel
        if not path.exists():
            raise AssertionError(f"{rel} is missing")


def _validate_required_text_markers(root: Path) -> None:
    for rel, markers in REQUIRED_TEXT_MARKERS.items():
        path = root / rel
        text = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            raise AssertionError(f"{rel} is missing required marker(s): {', '.join(missing)}")


def _validate_endings(root: Path) -> None:
    for path in _iter_files(root):
        suffix = path.suffix.lower()
        name = path.name.lower()
        if suffix in CRLF_SUFFIXES:
            _assert_line_ending(path, b"\r\n")
        elif suffix in LF_SUFFIXES or name in LF_NAMES:
            _assert_line_ending(path, b"\n")


def _validate_formats(root: Path) -> None:
    for path in _iter_files(root):
        suffix = path.suffix.lower()
        if suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".py":
            source = path.read_text(encoding="utf-8")
            ast.parse(source)
            compile(source, str(path), "exec")
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise AssertionError("PyYAML is required to validate workflow YAML") from exc
    workflow_dir = root / ".github" / "workflows"
    if workflow_dir.exists():
        for path in workflow_dir.glob("*.yml"):
            yaml.safe_load(path.read_text(encoding="utf-8"))


def _validate_requirements(root: Path) -> None:
    requirements = root / "requirements"
    if not requirements.exists():
        return
    for path in requirements.glob("*.txt"):
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]
        if not lines:
            raise AssertionError(f"{path} has no active requirement lines")
        for line in lines:
            if " " in line and not line.startswith(("--", "-")):
                raise AssertionError(f"{path} contains a space-separated requirement line: {line}")


def validate_root(root: Path) -> None:
    root = root.resolve()
    _validate_required_files(root)
    _validate_required_text_markers(root)
    _validate_endings(root)
    _validate_formats(root)
    _validate_requirements(root)


def _zip_root(extract_dir: Path) -> Path:
    children = [path for path in extract_dir.iterdir() if path.is_dir()]
    if len(children) == 1:
        return children[0]
    return extract_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT, help="Repository or extracted app root to validate.")
    parser.add_argument("--zip", type=Path, help="Release ZIP to extract and validate.")
    args = parser.parse_args()
    try:
        if args.zip:
            temp = Path(tempfile.mkdtemp(prefix="easy-asr-physical-"))
            try:
                with zipfile.ZipFile(args.zip) as archive:
                    archive.extractall(temp)
                validate_root(_zip_root(temp))
            finally:
                shutil.rmtree(temp, ignore_errors=True)
        else:
            validate_root(args.repo)
    except Exception as exc:
        print(f"physical file validation failed: {exc}", file=sys.stderr)
        return 1
    print("physical file validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
