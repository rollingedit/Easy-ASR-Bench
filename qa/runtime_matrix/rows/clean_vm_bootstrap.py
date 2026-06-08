from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from qa.runtime_matrix.common import ROOT, write_row
from qa.runtime_matrix.rows import setup_environment


PROOF_ENV = "EASY_ASR_BENCH_CLEAN_VM_BOOTSTRAP_PROOF"
SANDBOX_LAUNCH_ENV = "EASY_ASR_BENCH_LAUNCH_WINDOWS_SANDBOX"
SANDBOX_ROW_ID = "windows_sandbox_clean_bootstrap_deploy"


def _sandbox_script_text() -> str:
    return r'''$ErrorActionPreference = "Stop"
$Repo = "C:\Users\WDAGUtilityAccount\Desktop\Easy-ASR-Bench"
$Evidence = Join-Path $Repo "Temp\windows_sandbox_clean_bootstrap_evidence"
New-Item -ItemType Directory -Force -Path $Evidence | Out-Null
Set-Location $Repo

function Invoke-ProbeCommand {
  param([string[]]$Command)
  $resolved = Get-Command $Command[0] -ErrorAction SilentlyContinue
  $output = ""
  $exitCode = $null
  if ($resolved) {
    try {
      $exe = $Command[0]
      $args = @()
      if ($Command.Length -gt 1) {
        $args = $Command[1..($Command.Length - 1)]
      }
      $output = (& $exe @args 2>&1 | Out-String).Trim()
      $exitCode = $LASTEXITCODE
    } catch {
      $output = $_.Exception.Message
      $exitCode = -1
    }
  }
  [ordered]@{
    command = $Command
    resolved = if ($resolved) { $resolved.Source } else { "" }
    exit_code = $exitCode
    output = $output
    reports_python = [bool]($resolved -and $exitCode -eq 0 -and $output -match "Python \d+\.\d+")
  }
}

$probeCommands = @(
  @("python", "--version"),
  @("py", "--version"),
  @("py", "-3.12", "--version")
)
$probeResults = @()
foreach ($command in $probeCommands) {
  $probeResults += Invoke-ProbeCommand -Command $command
}
$preProbe = [ordered]@{
  schema = "easy_asr_bench.prebootstrap_python_probe.v1"
  system = "Windows"
  release = "11"
  python_visible_on_path = [bool]($probeResults | Where-Object { $_.reports_python })
  path_python_commands = $probeResults
}
$preProbePath = Join-Path $Evidence "prebootstrap-python-probe.json"
$preProbe | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -Path $preProbePath

$env:EASY_ASR_BENCH_PREBOOTSTRAP_PROBE = $preProbePath
$env:EASY_ASR_BENCH_CLEAN_VM_BOOTSTRAP_PROOF = "1"

$setupRepairLog = Join-Path $Evidence "setup-repair-all-safe.log"
$modelRepairLog = Join-Path $Evidence "setup-repair-model-layouts.log"
$win11RowLog = Join-Path $Evidence "win11-clean-no-python-row.log"
$cleanVmRowLog = Join-Path $Evidence "clean-vm-bootstrap-row.log"
$writeSmokeLog = Join-Path $Evidence "write-release-smoke.log"
$mergeSmokeLog = Join-Path $Evidence "merge-release-evidence.log"
$validateSmokeLog = Join-Path $Evidence "validate-release-smoke-evidence.log"
cmd /c setup.bat --doctor --repair-all-safe *> $setupRepairLog
cmd /c setup.bat --doctor --repair-model-layouts --allow-downloads *> $modelRepairLog
python qa\runtime_matrix\run_row.py --row win11_clean_no_python_setup --workdir Temp\windows_sandbox_clean_bootstrap_evidence *> $win11RowLog
python qa\runtime_matrix\run_row.py --row clean_vm_zero_dependency_bootstrap --workdir Temp\windows_sandbox_clean_bootstrap_evidence --install-deps --allow-downloads *> $cleanVmRowLog
python scripts\write_release_smoke.py --tag v0.3.9 --output release-smoke-v0.3.9-sandbox.json *> $writeSmokeLog
python scripts\merge_release_evidence.py --smoke release-smoke-v0.3.9-sandbox.json --evidence-dir Temp --output release-smoke-v0.3.9-sandbox.json --ignore-unknown *> $mergeSmokeLog
python scripts\validate_release_smoke.py --smoke release-smoke-v0.3.9-sandbox.json --required tests\fixtures\release_required_rows_v2.json --require-log-hashes --require-environment-summary *> $validateSmokeLog
'''


def _sandbox_config_text(script_path: Path) -> str:
    host_folder = escape(str(ROOT.resolve()))
    script_relative = script_path.resolve().relative_to(ROOT.resolve())
    sandbox_script = escape(r"C:\Users\WDAGUtilityAccount\Desktop\Easy-ASR-Bench" + "\\" + script_relative.as_posix().replace("/", "\\"))
    return (
        "<Configuration>\n"
        "  <MappedFolders>\n"
        "    <MappedFolder>\n"
        f"      <HostFolder>{host_folder}</HostFolder>\n"
        "      <SandboxFolder>C:\\Users\\WDAGUtilityAccount\\Desktop\\Easy-ASR-Bench</SandboxFolder>\n"
        "      <ReadOnly>false</ReadOnly>\n"
        "    </MappedFolder>\n"
        "  </MappedFolders>\n"
        "  <LogonCommand>\n"
        f"    <Command>powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"{sandbox_script}\"</Command>\n"
        "  </LogonCommand>\n"
        "</Configuration>\n"
    )


def _write_windows_sandbox_bundle(evidence_dir: Path) -> dict:
    script_path = evidence_dir / "Start-CleanVmValidation.ps1"
    config_path = evidence_dir / "Easy-ASR-Bench-Clean-Validation.wsb"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(_sandbox_script_text(), encoding="utf-8", newline="\r\n")
    config_path.write_text(_sandbox_config_text(script_path), encoding="utf-8", newline="\n")
    return {"script_path": str(script_path), "config_path": str(config_path)}


def _run_command(command: list[str], evidence_dir: Path, name: str, timeout: int = 3600) -> dict:
    transcript = evidence_dir / f"{name}.json"
    try:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        result = {
            "command": command,
            "exit_code": completed.returncode,
            "stdout_tail": completed.stdout[-6000:],
            "stderr_tail": completed.stderr[-6000:],
        }
    except Exception as exc:
        result = {
            "command": command,
            "exit_code": -1,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "stdout_tail": "",
            "stderr_tail": "",
        }
    transcript.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def _row_payload(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "missing", "read_error": f"{type(exc).__name__}: {exc}"}


def _json_from_command_stdout(result: dict) -> dict:
    try:
        return json.loads(str(result.get("stdout_tail") or ""))
    except json.JSONDecodeError as exc:
        return {"schema": "invalid", "read_error": f"JSONDecodeError: {exc}"}


def _setup_repair_evidence_failures(payload: dict) -> list[str]:
    failures: list[str] = []
    summary = payload.get("details", {}).get("repair_summary", {})
    for key in [
        "runtime_resolutions",
        "cached_runtime_resolutions",
        "previous_runtime_resolution_valid",
        "previous_runtime_resolution_stale",
        "backend_probes",
        "backend_probe_failed",
    ]:
        if key not in summary:
            failures.append(f"setup_repair_all_safe evidence missing repair_summary.{key}")
    if int(summary.get("runtime_resolutions", 0) or 0) <= 0:
        failures.append("setup_repair_all_safe evidence did not record any runtime resolutions")
    if not summary.get("repair_evidence_path"):
        failures.append("setup_repair_all_safe evidence missing repair_evidence_path")
    return failures


def _model_layout_repair_evidence_failures(payload: dict) -> list[str]:
    failures: list[str] = []
    details = payload.get("details", {})
    summary = details.get("sweep_summary", {})
    last_execution = details.get("last_execution_summary", {})
    for key in ["plan_count", "repaired", "blocked", "failed", "downloaded_files"]:
        if key not in summary:
            failures.append(f"setup_repair_model_layouts evidence missing sweep_summary.{key}")
    if int(summary.get("repaired", 0) or 0) <= 0:
        failures.append("setup_repair_model_layouts evidence did not repair any persisted model-layout plan")
    if int(summary.get("downloaded_files", 0) or 0) <= 0:
        failures.append("setup_repair_model_layouts evidence did not record any downloaded sidecars")
    if int(summary.get("blocked", 0) or 0) != 0 or int(summary.get("failed", 0) or 0) != 0:
        failures.append("setup_repair_model_layouts evidence reported blocked or failed repairs")
    if int(last_execution.get("repaired", 0) or 0) <= 0:
        failures.append("setup_repair_model_layouts evidence missing persisted last_execution repair")
    return failures


def _same_media_evidence_failures(payload: dict) -> list[str]:
    failures: list[str] = []
    details = payload.get("details", {})
    required_adapters = {
        "faster_whisper",
        "openai_whisper_pt",
        "generic_onnx_manifest",
        "hf_whisper_asr",
        "hf_transformers_asr",
        "whisper_cpp",
        "gguf_asr_mmproj",
    }
    run_adapters = set(details.get("run_adapters") or details.get("selected_adapters") or [])
    missing_adapters = sorted(required_adapters - run_adapters)
    if missing_adapters:
        failures.append("same-media benchmark evidence missing adapters: " + ", ".join(missing_adapters))
    if int(details.get("score_count", 0) or 0) < len(required_adapters):
        failures.append("same-media benchmark evidence did not score every required adapter")
    dependency_summary = details.get("dependency_resolution_summary", {})
    if dependency_summary.get("schema") != "easy_asr_bench.dependency_resolution_environment.v1":
        failures.append("same-media benchmark evidence missing dependency-resolution report schema")
    if int(dependency_summary.get("invalid_resolution_files", 0) or 0) != 0:
        failures.append("same-media benchmark evidence reported invalid dependency-resolution files")
    dependency_groups = set(details.get("dependency_resolution_groups") or [])
    required_groups = {"python_packaging", "media_tools", "faster_whisper", "onnx", "transformers_cpu", "whisper_cpp", "openai_whisper", "llama_cpp", "llama_mtmd"}
    missing_groups = sorted(required_groups - dependency_groups)
    if missing_groups:
        failures.append("same-media benchmark evidence missing dependency-resolution groups: " + ", ".join(missing_groups))
    last_repair = details.get("last_repair_summary", {})
    for key in ["runtime_resolutions", "cached_runtime_resolutions", "previous_runtime_resolution_valid", "previous_runtime_resolution_stale"]:
        if key not in last_repair:
            failures.append(f"same-media benchmark evidence missing last_repair_summary.{key}")
    return failures


def run(row_id: str, evidence_dir: Path, install_deps: bool, allow_downloads: bool) -> dict:
    if row_id == SANDBOX_ROW_ID:
        return _run_windows_sandbox_deploy(row_id, evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    setup_probe = setup_environment.run("win11_clean_no_python_setup", evidence_dir / "win11_clean_no_python_setup", False, False)
    details = {
        "proof_env": PROOF_ENV,
        "proof_env_value": os.environ.get(PROOF_ENV, ""),
        "install_deps": install_deps,
        "allow_downloads": allow_downloads,
        "setup_probe": {
            "status": setup_probe.get("status"),
            "summary": setup_probe.get("summary"),
            "block_reason": setup_probe.get("block_reason", ""),
            "external_requirement": setup_probe.get("external_requirement", ""),
        },
        "required_sequence": [
            "cmd /c setup.bat --doctor --repair-all-safe",
            "cmd /c setup.bat --doctor --repair-model-layouts --allow-downloads",
            "python qa/runtime_matrix/run_row.py --row setup_repair_all_safe --install-deps",
            "python qa/runtime_matrix/run_row.py --row setup_repair_model_layouts --allow-downloads",
            "python qa/runtime_matrix/run_row.py --row same_media_multi_model_smollm_benchmark --install-deps --allow-downloads",
            "python -m app.main --first-run-smoke --json",
        ],
    }
    artifacts = [evidence_dir / "win11_clean_no_python_setup" / "row.json"]
    if os.environ.get(PROOF_ENV) != "1":
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Clean VM zero-dependency bootstrap proof requires an explicit clean-VM marker before running destructive/long install validation.",
            block_reason=f"{PROOF_ENV}=1 is not set",
            external_requirement=(
                "In a fresh Windows 11 VM/Sandbox with no project dependencies preinstalled, set "
                f"{PROOF_ENV}=1 and run python qa\\runtime_matrix\\run_row.py --row clean_vm_zero_dependency_bootstrap "
                "--install-deps --allow-downloads"
            ),
            details=details,
            artifacts=artifacts,
        )
    if not install_deps or not allow_downloads:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Clean VM bootstrap proof must be allowed to repair dependencies and download fixture models/media.",
            block_reason="row was not run with both --install-deps and --allow-downloads",
            external_requirement=(
                f"rerun with {PROOF_ENV}=1, --install-deps, and --allow-downloads on the clean VM/Sandbox"
            ),
            details=details,
            artifacts=artifacts,
        )

    setup_repair = _run_command(["cmd", "/c", "setup.bat", "--doctor", "--repair-all-safe"], evidence_dir, "setup_doctor_repair_all_safe", timeout=3600)
    model_layout_repair = _run_command(
        ["cmd", "/c", "setup.bat", "--doctor", "--repair-model-layouts", "--allow-downloads"],
        evidence_dir,
        "setup_doctor_repair_model_layouts",
        timeout=3600,
    )
    repair_row_cmd = [
        sys.executable,
        "qa/runtime_matrix/run_row.py",
        "--row",
        "setup_repair_all_safe",
        "--workdir",
        str(evidence_dir / "subrows"),
        "--install-deps",
    ]
    repair_row = _run_command(repair_row_cmd, evidence_dir, "setup_repair_all_safe_row", timeout=3600)
    model_layout_row_cmd = [
        sys.executable,
        "qa/runtime_matrix/run_row.py",
        "--row",
        "setup_repair_model_layouts",
        "--workdir",
        str(evidence_dir / "subrows"),
        "--allow-downloads",
    ]
    model_layout_row = _run_command(model_layout_row_cmd, evidence_dir, "setup_repair_model_layouts_row", timeout=3600)
    benchmark_cmd = [
        sys.executable,
        "qa/runtime_matrix/run_row.py",
        "--row",
        "same_media_multi_model_smollm_benchmark",
        "--workdir",
        str(evidence_dir / "subrows"),
        "--install-deps",
        "--allow-downloads",
    ]
    benchmark_row = _run_command(benchmark_cmd, evidence_dir, "same_media_multi_model_row", timeout=7200)
    first_run = _run_command([sys.executable, "-m", "app.main", "--first-run-smoke", "--json"], evidence_dir, "first_run_smoke", timeout=900)

    repair_payload = _row_payload(evidence_dir / "subrows" / "setup_repair_all_safe" / "row.json")
    model_layout_payload = _row_payload(evidence_dir / "subrows" / "setup_repair_model_layouts" / "row.json")
    benchmark_payload = _row_payload(evidence_dir / "subrows" / "same_media_multi_model_smollm_benchmark" / "row.json")
    first_run_payload = _json_from_command_stdout(first_run)
    failures = []
    if setup_repair["exit_code"] != 0:
        failures.append("setup.bat --doctor --repair-all-safe failed")
    if model_layout_repair["exit_code"] != 0:
        failures.append("setup.bat --doctor --repair-model-layouts --allow-downloads failed")
    if repair_row["exit_code"] != 0 or repair_payload.get("status") != "pass":
        failures.append("setup_repair_all_safe subrow did not pass")
    else:
        failures.extend(_setup_repair_evidence_failures(repair_payload))
    if model_layout_row["exit_code"] != 0 or model_layout_payload.get("status") != "pass":
        failures.append("setup_repair_model_layouts subrow did not pass")
    else:
        failures.extend(_model_layout_repair_evidence_failures(model_layout_payload))
    if benchmark_row["exit_code"] != 0 or benchmark_payload.get("status") != "pass":
        failures.append("same-media multi-model benchmark subrow did not pass")
    else:
        failures.extend(_same_media_evidence_failures(benchmark_payload))
    if first_run["exit_code"] != 0:
        failures.append("first-run smoke failed")
    if first_run_payload.get("schema") != "easy_asr_bench.first_run_smoke.v1":
        failures.append("first-run smoke did not emit the expected JSON schema")
    if first_run_payload.get("repair_plan_schema") != "easy_asr_bench.repair_plan.v1":
        failures.append("first-run smoke did not include repair-plan evidence")
    if not first_run_payload.get("repair_command"):
        failures.append("first-run smoke did not include the setup repair command")
    if first_run_payload.get("model_layout_repair_command") != "setup.bat --doctor --repair-model-layouts --allow-downloads":
        failures.append("first-run smoke did not include the model-layout repair command")

    details.update(
        {
            "setup_doctor_repair_all_safe": setup_repair,
            "setup_doctor_repair_model_layouts": model_layout_repair,
            "setup_repair_all_safe_row": repair_row,
            "setup_repair_model_layouts_row": model_layout_row,
            "same_media_multi_model_row": benchmark_row,
            "first_run_smoke": first_run,
            "first_run_smoke_payload": first_run_payload,
            "setup_repair_all_safe_status": repair_payload.get("status"),
            "setup_repair_model_layouts_status": model_layout_payload.get("status"),
            "same_media_multi_model_status": benchmark_payload.get("status"),
            "failures": failures,
        }
    )
    artifacts.extend(
        [
            evidence_dir / "setup_doctor_repair_all_safe.json",
            evidence_dir / "setup_doctor_repair_model_layouts.json",
            evidence_dir / "setup_repair_all_safe_row.json",
            evidence_dir / "setup_repair_model_layouts_row.json",
            evidence_dir / "same_media_multi_model_row.json",
            evidence_dir / "first_run_smoke.json",
            evidence_dir / "subrows" / "setup_repair_all_safe" / "row.json",
            evidence_dir / "subrows" / "setup_repair_model_layouts" / "row.json",
            evidence_dir / "subrows" / "same_media_multi_model_smollm_benchmark" / "row.json",
        ]
    )
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "Clean VM bootstrap repaired dependencies, ran first-run smoke, and completed the same-media multi-model SmolLM benchmark."
            if not failures
            else "Clean VM bootstrap validation failed."
        ),
        details=details,
        artifacts=artifacts,
    )


def _run_windows_sandbox_deploy(row_id: str, evidence_dir: Path) -> dict:
    bundle = _write_windows_sandbox_bundle(evidence_dir)
    artifacts = [Path(bundle["script_path"]), Path(bundle["config_path"])]
    details = {
        "launch_env": SANDBOX_LAUNCH_ENV,
        "launch_env_value": os.environ.get(SANDBOX_LAUNCH_ENV, ""),
        "proof_env": PROOF_ENV,
        "prebootstrap_probe_env": setup_environment.PREBOOTSTRAP_PROBE_ENV,
        "bundle": bundle,
        "expected_sandbox_evidence_dir": "Temp\\windows_sandbox_clean_bootstrap_evidence",
    }
    if sys.platform != "win32":
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Windows Sandbox deployment can only run on Windows.",
            block_reason=f"current platform is {sys.platform}",
            external_requirement="Run this row on Windows with Windows Sandbox enabled.",
            details=details,
            artifacts=artifacts,
        )
    if os.environ.get(SANDBOX_LAUNCH_ENV) != "1":
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Windows Sandbox clean-bootstrap deployment bundle was generated but launch was not requested.",
            block_reason=f"{SANDBOX_LAUNCH_ENV}=1 is not set",
            external_requirement=(
                f"Enable Windows Sandbox, set {SANDBOX_LAUNCH_ENV}=1, then run "
                "python qa\\runtime_matrix\\run_row.py --row windows_sandbox_clean_bootstrap_deploy "
                "--workdir Temp\\runtime_matrix_windows_sandbox_clean_bootstrap_deploy"
            ),
            details=details,
            artifacts=artifacts,
        )
    completed = subprocess.run(["cmd", "/c", "start", "", str(Path(bundle["config_path"]).resolve())], cwd=ROOT, text=True, capture_output=True, timeout=60)
    details["launch_command"] = ["cmd", "/c", "start", "", str(Path(bundle["config_path"]).resolve())]
    details["launch_exit_code"] = completed.returncode
    details["launch_stdout_tail"] = completed.stdout[-2000:]
    details["launch_stderr_tail"] = completed.stderr[-2000:]
    return write_row(
        row_id,
        "pass" if completed.returncode == 0 else "fail",
        evidence_dir,
        summary=(
            "Windows Sandbox clean-bootstrap validation was launched; collect sandbox row evidence from the mapped Temp folder when it finishes."
            if completed.returncode == 0
            else "Windows Sandbox launch command failed."
        ),
        details=details,
        artifacts=artifacts,
    )
