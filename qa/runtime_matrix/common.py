from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from subprocess import run as _subprocess_run


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RowDefinition:
    row_id: str
    module: str
    description: str
    required: bool = False
    hardware: str = "local"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def environment_summary() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
    }


def git_state(root: Path = ROOT) -> dict:
    try:
        commit = _subprocess_run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
        status = _subprocess_run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True).stdout
        return {
            "execution_git_commit": commit,
            "execution_git_dirty": bool(status.strip()),
            "execution_git_status": "dirty" if status.strip() else "clean",
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        fallback = {
            "execution_git_commit": "unknown",
            "execution_git_dirty": None,
            "execution_git_status": "unknown",
        }
        env_commit = os.environ.get("EASY_ASR_COMMIT")
        if env_commit:
            fallback["execution_git_commit"] = env_commit
        return fallback


def dependency_resolution_report_failures(results: dict, *, expected_groups: set[str] | None = None) -> tuple[list[str], dict]:
    environment = results.get("environment", {})
    dependency_environment = environment.get("dependency_resolution_environment")
    if not isinstance(dependency_environment, dict):
        return ["results environment missing dependency_resolution_environment"], {}
    summary = dependency_environment.get("summary", {})
    failures: list[str] = []
    if summary.get("schema") != "easy_asr_bench.dependency_resolution_environment.v1":
        failures.append("dependency_resolution_environment summary schema is missing or invalid")
    if int(summary.get("invalid_resolution_files", 0) or 0) != 0:
        failures.append("dependency_resolution_environment reported invalid resolution files")
    groups = {item.get("dependency_group", "") for item in dependency_environment.get("resolutions", []) if isinstance(item, dict)}
    missing_groups = sorted((expected_groups or set()) - groups)
    last_repair = dependency_environment.get("last_repair_all_safe")
    if last_repair:
        repair_summary = last_repair.get("summary", {})
        for key in ["runtime_resolutions", "cached_runtime_resolutions", "previous_runtime_resolution_valid", "previous_runtime_resolution_stale"]:
            if key not in repair_summary:
                failures.append(f"last repair evidence missing summary.{key}")
        if expected_groups and missing_groups:
            failures.append("dependency resolution report missing groups: " + ", ".join(missing_groups))
    return failures, {
        "dependency_resolution_summary": summary,
        "dependency_resolution_groups": sorted(groups),
        "last_repair_summary": (last_repair or {}).get("summary", {}) if isinstance(last_repair, dict) else {},
        "missing_dependency_resolution_groups": missing_groups,
    }


def run_command(command: list[str], *, cwd: Path = ROOT, timeout: int = 300) -> dict:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def write_row(
    row_id: str,
    status: str,
    evidence_dir: Path,
    *,
    summary: str,
    details: dict | None = None,
    artifacts: list[Path] | None = None,
    block_reason: str = "",
    external_requirement: str = "",
) -> dict:
    if status not in {"pass", "fail", "blocked"}:
        raise ValueError(f"invalid row status: {status}")
    artifact_rows = []
    for path in artifacts or []:
        if path.exists() and path.is_file():
            artifact_rows.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    row = {
        "schema": "easy_asr_bench.runtime_matrix.row.v1",
        "id": row_id,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "environment": environment_summary(),
        "details": details or {},
        "artifacts": artifact_rows,
    }
    row.update(git_state())
    if status == "blocked":
        row["block_reason"] = block_reason
        row["external_requirement"] = external_requirement
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "row.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8", newline="\n")
    return row


def blocked_row_runner(reason: str, requirement: str) -> Callable[[str, Path, bool, bool], dict]:
    def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary=reason,
            block_reason=reason,
            external_requirement=requirement,
        )

    return run
