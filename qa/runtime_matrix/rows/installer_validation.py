from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from qa.runtime_matrix.common import ROOT, write_row


VERSION = "v0.3.9"
ZIP_NAME = f"Easy-ASR-Bench-{VERSION}-win.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _text_bytes(rel: str) -> bytes:
    data = (ROOT / rel).read_bytes()
    if b"\0" in data:
        return data
    suffix = Path(rel).suffix.lower()
    name = Path(rel).name.lower()
    if suffix in {".bat", ".cmd", ".ps1"}:
        return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n").encode("utf-8")
    if suffix in {".py", ".json", ".md", ".toml", ".ini", ".yml", ".yaml", ".html", ".css", ".js", ".txt"} or name in {".gitattributes", ".gitignore", ".editorconfig", "license"}:
        return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return data


def _write_normalized_file(source_rel: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_text_bytes(source_rel))


def _git_files() -> list[str]:
    completed = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=True)
    return [line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if line.strip()]


def _stage_release_assets(asset_dir: Path) -> dict[str, Path]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "setup.bat", asset_dir / "setup.bat")
    shutil.copy2(ROOT / "installer" / "install.ps1", asset_dir / "install.ps1")
    zip_path = asset_dir / ZIP_NAME
    files = [rel for rel in _git_files() if rel != "installer/checksums.json" and (ROOT / rel).is_file()]
    fixed_time = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for rel in sorted(files):
            info = zipfile.ZipInfo((Path(f"Easy-ASR-Bench-{VERSION}") / rel).as_posix(), fixed_time)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 0
            info.create_version = 20
            info.extract_version = 20
            archive.writestr(info, _text_bytes(rel))
    manifest = {
        "schema": "easy_asr_bench.installer_manifest.v2",
        "tag": VERSION,
        "version": VERSION[1:],
        "app_zip": ZIP_NAME,
        "installer_asset": "install.ps1",
        "install_dir": "%LOCALAPPDATA%\\Easy-ASR-Bench",
        "entrypoints": ["setup.bat", "Run.bat", "Drop_Audio_Or_Folders_Here.bat", "Open_Latest_Report.bat"],
    }
    manifest_path = asset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    checksums = {
        "schema": "easy_asr_bench.checksums.v1",
        "version": VERSION[1:],
        "files": {
            ZIP_NAME: _sha256(zip_path),
            "setup.bat": _sha256(asset_dir / "setup.bat"),
            "install.ps1": _sha256(asset_dir / "install.ps1"),
            "manifest.json": _sha256(manifest_path),
        },
    }
    checksums_path = asset_dir / "checksums.json"
    checksums_path.write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {
        "setup": asset_dir / "setup.bat",
        "install": asset_dir / "install.ps1",
        "manifest": manifest_path,
        "checksums": checksums_path,
        "zip": zip_path,
    }


def _run(command: list[str], *, temp_dir: Path | None = None) -> dict:
    env = os.environ.copy()
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        env["TEMP"] = str(temp_dir.resolve())
        env["TMP"] = str(temp_dir.resolve())
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180, env=env)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-6000:],
        "stderr_tail": completed.stderr[-6000:],
    }


def _install_path_with_spaces(row_id: str, evidence_dir: Path) -> dict:
    install_dir = evidence_dir / "Install Path With Spaces"
    result = _run([
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "installer\\install.ps1",
        "-InstallDir",
        str(install_dir),
        "-Version",
        VERSION,
        "-DryRun",
    ], temp_dir=evidence_dir / "tmp")
    failures = []
    output = result["stdout_tail"] + result["stderr_tail"]
    if result["exit_code"] != 0:
        failures.append("installer dry-run failed")
    if str(install_dir) not in output:
        failures.append("install path with spaces was not echoed in dry-run plan")
    if "no files will be downloaded, moved, or deleted" not in output:
        failures.append("dry-run safety marker missing")
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary="Installer dry-run accepts an install path containing spaces without modifying files." if not failures else "Install path with spaces validation failed.",
        details={"install_dir": str(install_dir), "command": result, "failures": failures},
    )


def _verify_release(row_id: str, evidence_dir: Path) -> dict:
    assets = _stage_release_assets(evidence_dir / "assets")
    result = _run(["cmd", "/c", "setup.bat", "--dry-run", "--verify-release", "--asset-dir", str(assets["manifest"].parent)], temp_dir=evidence_dir / "tmp")
    output = result["stdout_tail"] + result["stderr_tail"]
    failures = []
    for marker in ["Verified installer/install.ps1 SHA256", "[OK] release tag pinned", "[OK] ZIP layout valid", "[OK] release physical files valid"]:
        if marker not in output:
            failures.append(f"missing verify-release marker: {marker}")
    if result["exit_code"] != 0:
        failures.append("setup verify-release dry-run failed")
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary="setup.bat --dry-run --verify-release validates staged release assets and ZIP layout." if not failures else "setup verify-release dry-run failed.",
        details={"asset_dir": str(assets["manifest"].parent), "command": result, "failures": failures},
        artifacts=list(assets.values()),
    )


def _tamper_installer(row_id: str, evidence_dir: Path) -> dict:
    assets = _stage_release_assets(evidence_dir / "assets")
    (assets["install"]).write_text("# tampered installer\n", encoding="utf-8", newline="\n")
    result = _run(["cmd", "/c", "setup.bat", "--dry-run", "--asset-dir", str(assets["manifest"].parent)], temp_dir=evidence_dir / "tmp")
    output = result["stdout_tail"] + result["stderr_tail"]
    failures = []
    if result["exit_code"] == 0:
        failures.append("tampered installer was accepted")
    for marker in ["Integrity check failed before execution", "No files were installed or modified"]:
        if marker not in output:
            failures.append(f"missing tamper marker: {marker}")
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary="Tampered installer asset fails SHA verification before execution." if not failures else "Tampered installer was not blocked correctly.",
        details={"asset_dir": str(assets["manifest"].parent), "command": result, "failures": failures},
        artifacts=[assets["install"]],
    )


def _bad_release_checksum(row_id: str, evidence_dir: Path) -> dict:
    assets = _stage_release_assets(evidence_dir / "assets")
    checksums = json.loads(assets["checksums"].read_text(encoding="utf-8"))
    checksums["files"][ZIP_NAME] = "sha256:" + ("0" * 64)
    assets["checksums"].write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8", newline="\n")
    result = _run(["cmd", "/c", "setup.bat", "--dry-run", "--verify-release", "--asset-dir", str(assets["manifest"].parent)], temp_dir=evidence_dir / "tmp")
    output = result["stdout_tail"] + result["stderr_tail"]
    failures = []
    if result["exit_code"] == 0:
        failures.append("bad release checksum was accepted")
    if "Checksum mismatch" not in output:
        failures.append("checksum mismatch marker missing")
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary="Bad staged release checksum fails verify-release before install activation." if not failures else "Bad release checksum was not blocked correctly.",
        details={"asset_dir": str(assets["manifest"].parent), "command": result, "failures": failures},
        artifacts=[assets["checksums"], assets["zip"]],
    )


def _setup_doctor_strict(row_id: str, evidence_dir: Path) -> dict:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    config_path = evidence_dir / "config.json"
    folders = {
        "models": str(evidence_dir / "Models"),
        "input": str(evidence_dir / "Input"),
        "output": str(evidence_dir / "Output"),
        "temp": str(evidence_dir / "Temp"),
        "logs": str(evidence_dir / "Logs"),
        "cache": str(evidence_dir / "Cache"),
    }
    config_path.write_text(json.dumps({"folders": folders}, indent=2) + "\n", encoding="utf-8", newline="\n")
    temp_dir = evidence_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TEMP"] = str(temp_dir.resolve())
    env["TMP"] = str(temp_dir.resolve())
    command = [sys.executable, "-m", "app.doctor", "--config", str(config_path), "--strict", "--json"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180, env=env)
    result = {
        "command": command,
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-6000:],
        "stderr_tail": completed.stderr[-6000:],
    }
    report_path = evidence_dir / "doctor.json"
    failures = []
    report = {}
    if result["exit_code"] != 0:
        failures.append("strict doctor exited nonzero")
    try:
        report = json.loads(completed.stdout)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    except json.JSONDecodeError as exc:
        failures.append(f"doctor JSON could not be parsed: {exc}")
    if report.get("schema") != "easy_asr_bench.doctor.v1":
        failures.append("doctor schema marker missing")
    if not report.get("dependency_status", {}).get("core", {}).get("available"):
        failures.append("doctor strict did not prove core dependency availability")
    if "cuda_provider_checks" not in report:
        failures.append("doctor provider diagnostics missing")
    hf_cache = report.get("cuda_provider_checks", {}).get("huggingface_cache", {})
    if not hf_cache:
        failures.append("doctor Hugging Face cache diagnostic missing")
    elif not hf_cache.get("cache_dir"):
        failures.append("doctor Hugging Face cache directory missing")
    for folder in folders.values():
        if not Path(folder).exists():
            failures.append(f"doctor did not create/check folder {folder}")
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary="Strict doctor JSON reports core dependency status, provider diagnostics, and isolated folder paths." if not failures else "Strict doctor validation failed.",
        details={
            "config_path": str(config_path),
            "command": result,
            "core_available": report.get("dependency_status", {}).get("core", {}).get("available"),
            "dependency_groups": sorted(report.get("dependency_status", {})),
            "huggingface_cache": hf_cache,
            "failures": failures,
        },
        artifacts=[config_path, report_path],
    )


def _write_isolated_config(evidence_dir: Path) -> tuple[Path, dict[str, str]]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    config_path = evidence_dir / "config.json"
    folders = {
        "models": str(evidence_dir / "Models"),
        "input": str(evidence_dir / "Input"),
        "output": str(evidence_dir / "Output"),
        "temp": str(evidence_dir / "Temp"),
        "logs": str(evidence_dir / "Logs"),
        "cache": str(evidence_dir / "Cache"),
    }
    config_path.write_text(json.dumps({"folders": folders}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return config_path, folders


def _run_python_doctor(command_args: list[str], evidence_dir: Path) -> tuple[dict, subprocess.CompletedProcess[str]]:
    temp_dir = evidence_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TEMP"] = str(temp_dir.resolve())
    env["TMP"] = str(temp_dir.resolve())
    command = [sys.executable, "-m", "app.doctor", *command_args]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=240, env=env)
    return (
        {
            "command": command,
            "exit_code": completed.returncode,
            "stdout_tail": completed.stdout[-6000:],
            "stderr_tail": completed.stderr[-6000:],
        },
        completed,
    )


def _setup_repair_all_safe(row_id: str, evidence_dir: Path, install_deps: bool) -> dict:
    config_path, folders = _write_isolated_config(evidence_dir)
    plan_result, plan_completed = _run_python_doctor(["--config", str(config_path), "--repair-plan"], evidence_dir)
    plan_path = evidence_dir / "repair_plan.json"
    repair_path = evidence_dir / "repair_all_safe.json"
    failures = []
    plan = {}
    try:
        plan = json.loads(plan_completed.stdout)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8", newline="\n")
    except json.JSONDecodeError as exc:
        failures.append(f"repair-plan JSON could not be parsed: {exc}")
    if plan_result["exit_code"] != 0:
        failures.append("repair-plan exited nonzero")
    if plan.get("schema") != "easy_asr_bench.repair_plan.v1":
        failures.append("repair-plan schema marker missing")
    needs_repair = int(plan.get("summary", {}).get("needs_repair", 0) or 0)
    can_auto_repair = int(plan.get("summary", {}).get("can_auto_repair", 0) or 0)
    if failures:
        return write_row(
            row_id,
            "fail",
            evidence_dir,
            summary="setup repair-all-safe preflight failed before repair execution.",
            details={"config_path": str(config_path), "folders": folders, "plan_command": plan_result, "failures": failures},
            artifacts=[config_path, plan_path],
        )
    if needs_repair and not install_deps:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="repair-all-safe preflight found missing repairable dependencies, but --install-deps was not allowed.",
            block_reason=f"repair plan needs {needs_repair} dependency repair(s), {can_auto_repair} auto-repairable",
            external_requirement="Rerun with python qa\\runtime_matrix\\run_row.py --row setup_repair_all_safe --install-deps to exercise safe repair execution.",
            details={"config_path": str(config_path), "folders": folders, "plan_summary": plan.get("summary", {}), "plan_command": plan_result},
            artifacts=[config_path, plan_path],
        )
    repair_result, repair_completed = _run_python_doctor(["--config", str(config_path), "--repair-all-safe"], evidence_dir)
    repair = {}
    try:
        repair = json.loads(repair_completed.stdout)
        repair_path.write_text(json.dumps(repair, indent=2) + "\n", encoding="utf-8", newline="\n")
    except json.JSONDecodeError as exc:
        failures.append(f"repair-all-safe JSON could not be parsed: {exc}")
    summary = repair.get("summary", {})
    if repair_result["exit_code"] != 0:
        failures.append("repair-all-safe exited nonzero")
    if repair.get("schema") != "easy_asr_bench.repair_plan.v1":
        failures.append("repair-all-safe schema marker missing")
    if repair.get("mode") != "repair_all_safe":
        failures.append("repair-all-safe mode marker missing")
    if int(summary.get("backend_probes", 0) or 0) <= 0:
        failures.append("backend probe summary count missing")
    if int(summary.get("runtime_resolutions", 0) or 0) <= 0:
        failures.append("runtime resolution summary count missing")
    if "cached_runtime_resolutions" not in summary:
        failures.append("cached runtime resolution count missing")
    if "previous_runtime_resolution_valid" not in summary:
        failures.append("previous runtime resolution validity count missing")
    if "previous_runtime_resolution_stale" not in summary:
        failures.append("previous runtime resolution stale count missing")
    if int(summary.get("backend_probe_failed", 0) or 0) != 0:
        failures.append("one or more dependency backends failed after repair")
    records = repair.get("records", [])
    if not all(record.get("after", {}).get("backend_probe") for record in records if record.get("after", {}).get("repair_result") in {"already_ok", "repaired"}):
        failures.append("one or more repaired/ok records are missing backend_probe evidence")
    if not all(record.get("after", {}).get("runtime_resolution_path") for record in records if record.get("after", {}).get("repair_result") in {"already_ok", "repaired"} and record.get("after", {}).get("backend_probe", {}).get("ok")):
        failures.append("one or more usable dependency backends are missing runtime_resolution_path evidence")
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary="repair-all-safe emits structured after-state dependency and backend-probe evidence." if not failures else "repair-all-safe validation failed.",
        details={
            "config_path": str(config_path),
            "folders": folders,
            "plan_summary": plan.get("summary", {}),
            "repair_summary": summary,
            "plan_command": plan_result,
            "repair_command": repair_result,
            "install_deps_allowed": install_deps,
            "failures": failures,
        },
        artifacts=[config_path, plan_path, repair_path],
    )


def _write_fake_install_tree(install_dir: Path) -> dict[str, Path]:
    runtime_paths = [
        install_dir / "app" / "main.py",
        install_dir / "requirements" / "core.txt",
        install_dir / "scripts" / "validate_release_files.py",
        install_dir / "installer" / "install.ps1",
        install_dir / ".github" / "workflows" / "release-gate.yml",
        install_dir / ".venv" / "pyvenv.cfg",
        install_dir / "Run.bat",
        install_dir / "setup.bat",
        install_dir / "README.md",
    ]
    user_paths = {
        "Models": install_dir / "Models" / "keep-model.txt",
        "Input": install_dir / "Input" / "keep-input.wav",
        "Output": install_dir / "Output" / "keep-output.txt",
        "Logs": install_dir / "Logs" / "keep-log.txt",
        "Cache": install_dir / "Cache" / "keep-cache.txt",
        "Temp": install_dir / "Temp" / "keep-temp.txt",
        "config": install_dir / "config.json",
    }
    for path in runtime_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("runtime fixture\n", encoding="utf-8", newline="\n")
    for path in user_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("user data fixture\n", encoding="utf-8", newline="\n")
    return user_paths


def _extract_preservation_functions() -> str:
    text = (ROOT / "installer" / "install.ps1").read_text(encoding="utf-8")
    match = re.search(
        r"function Get-TreeStats\(\$Path\) \{.*?^if \(\$Doctor\) \{",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise RuntimeError("could not extract installer preservation functions")
    return match.group(0).removesuffix('if ($Doctor) {').rstrip()


def _ps_literal(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _write_fake_new_install_tree(new_dir: Path) -> None:
    for path in [
        new_dir / "app" / "main.py",
        new_dir / "requirements" / "core.txt",
        new_dir / "Run.bat",
        new_dir / "Models" / "default-model-placeholder.txt",
        new_dir / "config.json",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new install fixture\n", encoding="utf-8", newline="\n")


def _preservation_harness_script(old_dir: Path, new_dir: Path, output_path: Path, *, restore: bool) -> str:
    functions = _extract_preservation_functions()
    restore_call = "Restore-MovedUserData $NewInstall $OldInstall" if restore else ""
    phase = "after_restore" if restore else "after_move"
    return f"""
$ErrorActionPreference = "Stop"
{functions}
$OldInstall = {_ps_literal(old_dir)}
$NewInstall = {_ps_literal(new_dir)}
Move-PreservedUserData $OldInstall $NewInstall
$reportPath = Join-Path $NewInstall "Logs\\install-preservation-report.json"
$report = $null
if (Test-Path $reportPath) {{
  $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
}}
$reportWasWritten = Test-Path $reportPath
{restore_call}
$names = @("Models", "Input", "Output", "Logs", "Cache", "Temp", "config.json")
$states = @{{}}
foreach ($name in $names) {{
  $states[$name] = @{{
    old_exists = Test-Path (Join-Path $OldInstall $name)
    new_exists = Test-Path (Join-Path $NewInstall $name)
  }}
}}
$result = [ordered]@{{
  phase = "{phase}"
  preservation_report_exists = $reportWasWritten
  preservation_report_exists_after_phase = Test-Path $reportPath
  preservation_report_schema = if ($report) {{ $report.schema }} else {{ "" }}
  item_status = @($report.items | ForEach-Object {{ [ordered]@{{ name = $_.name; status = $_.status; file_count = $_.file_count }} }})
  states = $states
}}
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath {_ps_literal(output_path)} -Encoding UTF8
"""


def _run_preservation_harness(evidence_dir: Path, *, restore: bool) -> tuple[dict, Path, Path, Path]:
    old_dir = evidence_dir / "Existing Install.backup"
    new_dir = evidence_dir / "New Install"
    output_path = evidence_dir / ("rollback_restore.json" if restore else "update_preservation.json")
    script_path = evidence_dir / ("invoke_rollback_restore.ps1" if restore else "invoke_update_preservation.ps1")
    _write_fake_install_tree(old_dir)
    _write_fake_new_install_tree(new_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text(_preservation_harness_script(old_dir, new_dir, output_path, restore=restore), encoding="utf-8", newline="\n")
    result = _run([
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
    ], temp_dir=evidence_dir / "tmp")
    return result, output_path, old_dir, new_dir


def _update_preserves_user_data(row_id: str, evidence_dir: Path) -> dict:
    result, output_path, old_dir, new_dir = _run_preservation_harness(evidence_dir, restore=False)
    details = {}
    failures = []
    if result["exit_code"] != 0:
        failures.append("preservation harness exited nonzero")
    if output_path.exists():
        details = json.loads(output_path.read_text(encoding="utf-8-sig"))
    else:
        failures.append("preservation harness did not write JSON evidence")
    if details.get("preservation_report_schema") != "easy_asr_bench.install_preservation_report.v1":
        failures.append("preservation report schema missing")
    states = details.get("states", {})
    for name in ["Models", "Input", "Output", "Logs", "Cache", "Temp", "config.json"]:
        state = states.get(name, {})
        if state.get("old_exists"):
            failures.append(f"user data still exists in old install after move: {name}")
        if not state.get("new_exists"):
            failures.append(f"user data was not moved into new install: {name}")
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary="Installer preservation functions move user data into the new install without copying large folders." if not failures else "Update preservation harness failed.",
        details={"command": result, "old_dir": str(old_dir), "new_dir": str(new_dir), "harness": details, "failures": failures},
        artifacts=[output_path, new_dir / "Logs" / "install-preservation-report.json"],
    )


def _interrupted_download_rollback(row_id: str, evidence_dir: Path) -> dict:
    result, output_path, old_dir, new_dir = _run_preservation_harness(evidence_dir, restore=True)
    details = {}
    failures = []
    if result["exit_code"] != 0:
        failures.append("rollback harness exited nonzero")
    if output_path.exists():
        details = json.loads(output_path.read_text(encoding="utf-8-sig"))
    else:
        failures.append("rollback harness did not write JSON evidence")
    states = details.get("states", {})
    for name in ["Models", "Input", "Output", "Logs", "Cache", "Temp", "config.json"]:
        state = states.get(name, {})
        if not state.get("old_exists"):
            failures.append(f"user data was not restored to backup install: {name}")
        if state.get("new_exists"):
            failures.append(f"user data remained in failed new install after rollback: {name}")
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary="Installer rollback functions restore moved user data to the previous install when activation fails." if not failures else "Rollback preservation harness failed.",
        details={"command": result, "old_dir": str(old_dir), "new_dir": str(new_dir), "harness": details, "failures": failures},
        artifacts=[output_path],
    )


def _repair_broken_venv(row_id: str, evidence_dir: Path) -> dict:
    install_dir = evidence_dir / "Broken Venv Install"
    _write_fake_install_tree(install_dir)
    broken_cfg = install_dir / ".venv" / "pyvenv.cfg"
    broken_cfg.parent.mkdir(parents=True, exist_ok=True)
    broken_cfg.write_text("home = missing\n", encoding="utf-8", newline="\n")
    venv_python = install_dir / ".venv" / "Scripts" / "python.exe"
    result = _run([
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "installer\\install.ps1",
        "-InstallDir",
        str(install_dir),
        "-Version",
        VERSION,
        "-DryRun",
        "-Repair",
    ], temp_dir=evidence_dir / "tmp")
    output = result["stdout_tail"] + result["stderr_tail"]
    failures = []
    if result["exit_code"] != 0:
        failures.append("repair dry-run exited nonzero")
    for marker in [
        "repair mode: runtime files and .venv will be refreshed after release verification",
        "detected broken venv: .venv\\Scripts\\python.exe is missing",
        "preserved user data: Models, Input, Output, Logs, Cache, Temp, config.json",
        "no files will be downloaded, moved, or deleted",
    ]:
        if marker not in output:
            failures.append(f"missing repair dry-run marker: {marker}")
    if venv_python.exists():
        failures.append("repair dry-run modified the fake broken venv")
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary="Repair dry-run detects a broken virtualenv and promises a verified runtime refresh while preserving user data." if not failures else "Broken venv repair dry-run validation failed.",
        details={"install_dir": str(install_dir), "command": result, "failures": failures},
        artifacts=[broken_cfg],
    )


def _uninstall_preserve(row_id: str, evidence_dir: Path) -> dict:
    install_dir = evidence_dir / "Fake Install"
    user_paths = _write_fake_install_tree(install_dir)
    result = _run([
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "installer\\install.ps1",
        "-InstallDir",
        str(install_dir),
        "-Uninstall",
    ], temp_dir=evidence_dir / "tmp")
    failures = []
    if result["exit_code"] != 0:
        failures.append("preserving uninstall exited nonzero")
    for name, path in user_paths.items():
        if not path.exists():
            failures.append(f"user data was not preserved: {name}")
    for path in [install_dir / "app", install_dir / ".venv", install_dir / "Run.bat", install_dir / "setup.bat"]:
        if path.exists():
            failures.append(f"runtime path was not removed: {path.name}")
    if "User data was preserved" not in result["stdout_tail"]:
        failures.append("preservation message missing")
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary="Uninstall removes app/runtime files while preserving Models/Input/Output/Logs/Cache/Temp/config.json." if not failures else "Preserving uninstall validation failed.",
        details={"install_dir": str(install_dir), "command": result, "failures": failures},
        artifacts=[path for path in user_paths.values() if path.exists()],
    )


def _destructive_uninstall_requires_phrase(row_id: str, evidence_dir: Path) -> dict:
    install_dir = evidence_dir / "Fake Install"
    user_paths = _write_fake_install_tree(install_dir)
    result = _run([
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "installer\\install.ps1",
        "-InstallDir",
        str(install_dir),
        "-Uninstall",
        "-RemoveUserData",
    ], temp_dir=evidence_dir / "tmp")
    output = result["stdout_tail"] + result["stderr_tail"]
    failures = []
    if result["exit_code"] == 0:
        failures.append("destructive uninstall without phrase exited zero")
    if "DELETE EASY ASR BENCH USER DATA" not in output:
        failures.append("required confirmation phrase was not shown")
    for name, path in user_paths.items():
        if not path.exists():
            failures.append(f"user data was removed despite missing phrase: {name}")
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary="Destructive uninstall refuses to remove user data unless the exact confirmation phrase is supplied." if not failures else "Destructive uninstall phrase guard failed.",
        details={"install_dir": str(install_dir), "command": result, "failures": failures},
        artifacts=[path for path in user_paths.values() if path.exists()],
    )


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id == "install_path_with_spaces":
        return _install_path_with_spaces(row_id, evidence_dir)
    if row_id == "setup_dry_run_verify_release":
        return _verify_release(row_id, evidence_dir)
    if row_id in {"tampered_installer_fails_before_execution", "bad_checksum_fails_before_execution"}:
        return _tamper_installer(row_id, evidence_dir)
    if row_id == "setup_verify_release_bad_checksum":
        return _bad_release_checksum(row_id, evidence_dir)
    if row_id == "setup_doctor_strict":
        return _setup_doctor_strict(row_id, evidence_dir)
    if row_id == "setup_repair_all_safe":
        return _setup_repair_all_safe(row_id, evidence_dir, _install_deps)
    if row_id == "update_preserves_user_data":
        return _update_preserves_user_data(row_id, evidence_dir)
    if row_id == "interrupted_download_rollback":
        return _interrupted_download_rollback(row_id, evidence_dir)
    if row_id in {"repair_broken_venv", "broken_venv_repair"}:
        return _repair_broken_venv(row_id, evidence_dir)
    if row_id == "uninstall_preserve_user_data":
        return _uninstall_preserve(row_id, evidence_dir)
    if row_id == "destructive_uninstall_requires_phrase":
        return _destructive_uninstall_requires_phrase(row_id, evidence_dir)
    return write_row(row_id, "fail", evidence_dir, summary=f"Unsupported installer validation row: {row_id}")
