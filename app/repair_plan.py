from __future__ import annotations

import json
import importlib.metadata
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .ctranslate2_probe import ctranslate2_probe
from .dependency_manager import acceleration_install_decision, dependency_status, install_group_for_config, missing_modules_for_config


GROUP_IMPORT_PROBES = {
    "python_packaging": ("pip", "setuptools", "pkg_resources"),
    "core": ("jinja2", "numpy", "soundfile", "jiwer"),
    "transformers_cpu": ("torch", "transformers", "safetensors", "sentencepiece", "google.protobuf", "torchaudio"),
    "openai_whisper": ("whisper", "torch"),
    "whisper_cpp": ("pywhispercpp.model",),
    "llama_cpp": ("llama_cpp",),
}


def _windows_error_mode_prefix() -> str:
    return (
        "import sys\n"
        "if sys.platform == 'win32':\n"
        "    import ctypes\n"
        "    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)\n"
    )


def _isolated_import_probe(modules: tuple[str, ...], timeout: int = 30) -> dict:
    payload = {"modules": list(modules)}
    code = (
        _windows_error_mode_prefix()
        + """
import importlib
import json
import sys

payload = json.loads(sys.argv[1])
loaded = []
try:
    for module in payload["modules"]:
        imported = importlib.import_module(module)
        loaded.append({"module": module, "version": getattr(imported, "__version__", "")})
    print(json.dumps({"ok": True, "loaded": loaded}))
except BaseException as exc:
    print(json.dumps({"ok": False, "loaded": loaded, "error_type": type(exc).__name__, "error": str(exc)}))
    raise SystemExit(1)
"""
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code, json.dumps(payload)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "loaded": [], "error_type": type(exc).__name__, "error": str(exc)}
    stdout = completed.stdout.strip()
    try:
        result = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        detail = (completed.stderr or stdout or f"exit {completed.returncode}").strip()
        return {"ok": False, "loaded": [], "error_type": "ImportProbeFailed", "error": detail}
    if completed.returncode == 0 and result.get("ok"):
        return result
    result["ok"] = False
    if "returncode" not in result:
        result["returncode"] = completed.returncode
    if completed.stderr and "stderr" not in result:
        result["stderr"] = completed.stderr.strip()[-2000:]
    return result


def _isolated_whisper_cpp_probe() -> dict:
    code = (
        _windows_error_mode_prefix()
        + """
import json
try:
    from pywhispercpp.model import Model
    transcribe_available = callable(getattr(Model, "transcribe", None))
    print(json.dumps({"ok": transcribe_available, "transcribe_available": transcribe_available}))
except BaseException as exc:
    print(json.dumps({"ok": False, "transcribe_available": False, "error_type": type(exc).__name__, "error": str(exc)}))
    raise SystemExit(1)
"""
    )
    try:
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "transcribe_available": False, "error_type": type(exc).__name__, "error": str(exc)}
    stdout = completed.stdout.strip()
    try:
        result = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        detail = (completed.stderr or stdout or f"exit {completed.returncode}").strip()
        return {"ok": False, "transcribe_available": False, "error_type": "WhisperCppProbeFailed", "error": detail}
    if completed.returncode != 0:
        result["ok"] = False
        result["returncode"] = completed.returncode
    return result


def backend_probe_for_group(group: str, config: dict) -> dict:
    acceleration = acceleration_install_decision(config, group)
    if group == "onnx":
        from .dependency_manager import onnxruntime_available_providers

        providers, error = onnxruntime_available_providers()
        expected_provider = {
            "cuda": "CUDAExecutionProvider",
            "directml": "DmlExecutionProvider",
            "openvino": "OpenVINOExecutionProvider",
        }.get(str(acceleration.get("accelerator") or ""))
        expected_present = True if not expected_provider else expected_provider in providers
        return {
            "kind": "onnxruntime_provider_probe",
            "ok": bool(providers) and not error,
            "providers": providers,
            "expected_provider": expected_provider or "",
            "expected_provider_present": expected_present,
            "accelerator_probe": {
                "requested": bool(expected_provider),
                "accelerator": acceleration.get("accelerator") or "",
                "ok": expected_present,
                "reason": "" if expected_present else f"{expected_provider} was not listed by ONNX Runtime.",
            },
            "error": error,
        }
    if group == "faster_whisper":
        imports = _isolated_import_probe(("faster_whisper", "ctranslate2", "pkg_resources"))
        ctranslate2 = ctranslate2_probe()
        ok = bool(imports.get("ok")) and bool(ctranslate2.get("ok"))
        accelerator_requested = acceleration.get("accelerator") == "cuda"
        accelerator_ok = not accelerator_requested or bool(ctranslate2.get("cuda_available"))
        return {
            "kind": "faster_whisper_ctranslate2_import_probe",
            "ok": ok,
            "imports": imports,
            "ctranslate2": ctranslate2,
            "expected_accelerator": acceleration.get("accelerator") or "",
            "accelerator_probe": {
                "requested": accelerator_requested,
                "accelerator": acceleration.get("accelerator") or "",
                "ok": accelerator_ok,
                "reason": "" if accelerator_ok else "CTranslate2 CUDA backend was not verified.",
            },
        }
    if group == "whisper_cpp":
        probe = _isolated_whisper_cpp_probe()
        return {"kind": "whisper_cpp_import_probe", "ok": bool(probe.get("ok")), "imports": probe}
    if group == "llama_cpp":
        from .dependency_manager import llama_cpp_gpu_capable

        imports = _isolated_import_probe(("llama_cpp",))
        gpu_required = acceleration.get("accelerator") in {"cuda", "vulkan"}
        gpu_capable = llama_cpp_gpu_capable()
        return {
            "kind": "llama_cpp_import_probe",
            "ok": bool(imports.get("ok")),
            "imports": imports,
            "expected_accelerator": acceleration.get("accelerator") or "",
            "gpu_offload_available": gpu_capable,
            "accelerator_probe": {
                "requested": gpu_required,
                "accelerator": acceleration.get("accelerator") or "",
                "ok": not gpu_required or gpu_capable,
                "reason": "" if (not gpu_required or gpu_capable) else "llama-cpp-python GPU/offload backend was not verified.",
            },
        }
    if group == "llama_mtmd":
        from .dependency_manager import llama_mtmd_cli_status

        status = llama_mtmd_cli_status(config)
        return {
            "kind": "llama_mtmd_runtime_probe",
            "ok": bool(status.get("available") or status.get("qwen3_asr_handler_available")),
            "runtime_status": status,
        }
    if group == "media_tools":
        from .dependency_manager import media_tools_status

        status = media_tools_status()
        return {
            "kind": "media_tools_ffmpeg_probe",
            "ok": bool(status.get("available")),
            "runtime_status": status,
        }
    modules = GROUP_IMPORT_PROBES.get(group)
    if modules:
        probe = _isolated_import_probe(modules)
        return {"kind": "python_import_probe", "ok": bool(probe.get("ok")), "imports": probe}
    return {"kind": "none", "ok": True, "message": "No backend probe is defined for this dependency group."}


def _probe_versions(probe: dict) -> dict[str, str]:
    versions: dict[str, str] = {}
    imports = probe.get("imports", {})
    for item in imports.get("loaded", []) if isinstance(imports, dict) else []:
        module = str(item.get("module") or "")
        version = str(item.get("version") or "")
        if module:
            versions[module] = version
    ctranslate2 = probe.get("ctranslate2", {})
    if isinstance(ctranslate2, dict) and ctranslate2.get("version"):
        versions["ctranslate2"] = str(ctranslate2["version"])
    return versions


def _probe_runtime_path(probe: dict) -> str:
    runtime_status = probe.get("runtime_status", {}) if isinstance(probe, dict) else {}
    if not isinstance(runtime_status, dict):
        return ""
    return str(runtime_status.get("path") or runtime_status.get("ffmpeg_path") or "")


def build_runtime_resolution(record: dict, config: dict, project_root: Path) -> dict:
    group = str(record["affected_dependency_group"])
    after = record.get("after", {})
    probe = after.get("backend_probe", {})
    accelerator_probe = probe.get("accelerator_probe", {}) if isinstance(probe, dict) else {}
    accelerator_requested = bool(accelerator_probe.get("requested", False))
    accelerator_verified = bool(accelerator_probe.get("ok", False)) if accelerator_requested else False
    backend_verified = bool(probe.get("ok", False)) if isinstance(probe, dict) else False
    status = "backend_usable"
    if accelerator_requested and accelerator_verified:
        status = "accelerator_verified"
    elif accelerator_requested and backend_verified:
        status = "backend_usable_accelerator_unverified"
    elif not backend_verified:
        status = "backend_unverified"
    return {
        "schema": "easy_asr_bench.runtime_resolution.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "dependency_group": group,
        "status": status,
        "repair_result": after.get("repair_result", ""),
        "backend_verified": backend_verified,
        "backend_probe_kind": probe.get("kind", "") if isinstance(probe, dict) else "",
        "accelerator_requested": accelerator_requested,
        "accelerator": accelerator_probe.get("accelerator", "") if isinstance(accelerator_probe, dict) else "",
        "accelerator_verified": accelerator_verified,
        "accelerator_reason": accelerator_probe.get("reason", "") if isinstance(accelerator_probe, dict) else "",
        "install_kind": record.get("before", {}).get("install_kind", ""),
        "requirement_file": record.get("before", {}).get("requirement_file", ""),
        "repair_command": record.get("repair_command", ""),
        "versions": _probe_versions(probe) if isinstance(probe, dict) else {},
        "providers": probe.get("providers", []) if isinstance(probe, dict) else [],
        "runtime_path": _probe_runtime_path(probe) if isinstance(probe, dict) else "",
        "config_runtime": {
            "provider": config.get("runtime", {}).get("provider", ""),
            "prefer_gpu": config.get("runtime", {}).get("prefer_gpu", ""),
            "fallback_to_cpu": config.get("runtime", {}).get("fallback_to_cpu", ""),
        },
    }


def validate_saved_runtime_resolution(saved: dict, current: dict) -> dict:
    checks = {
        "schema": saved.get("schema") == current.get("schema"),
        "dependency_group": saved.get("dependency_group") == current.get("dependency_group"),
        "config_runtime": saved.get("config_runtime") == current.get("config_runtime"),
        "backend_probe_kind": saved.get("backend_probe_kind") == current.get("backend_probe_kind"),
        "status": saved.get("status") == current.get("status"),
        "versions": saved.get("versions", {}) == current.get("versions", {}),
        "providers": saved.get("providers", []) == current.get("providers", []),
        "runtime_path": saved.get("runtime_path", "") == current.get("runtime_path", ""),
        "accelerator": saved.get("accelerator", "") == current.get("accelerator", ""),
        "accelerator_verified": saved.get("accelerator_verified", False) == current.get("accelerator_verified", False),
    }
    mismatches = [name for name, ok in checks.items() if not ok]
    return {
        "schema": "easy_asr_bench.runtime_resolution_check.v1",
        "status": "valid" if not mismatches else "stale",
        "mismatches": mismatches,
        "checks": checks,
    }


def persist_runtime_resolution(record: dict, config: dict, project_root: Path) -> dict:
    resolution = build_runtime_resolution(record, config, project_root)
    path = project_root / "Logs" / f"dependency_resolution_{record['affected_dependency_group']}.json"
    previous_check = {
        "schema": "easy_asr_bench.runtime_resolution_check.v1",
        "status": "missing",
        "mismatches": [],
        "checks": {},
    }
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            previous_check = validate_saved_runtime_resolution(previous, resolution)
        except json.JSONDecodeError as exc:
            previous_check = {
                "schema": "easy_asr_bench.runtime_resolution_check.v1",
                "status": "invalid_json",
                "mismatches": ["json"],
                "checks": {},
                "error": str(exc),
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resolution, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"path": str(path), "resolution": resolution, "previous_resolution_check": previous_check}


def persist_repair_plan_evidence(plan: dict, project_root: Path) -> str:
    path = project_root / "Logs" / "repair_all_safe_last.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    plan["summary"]["repair_evidence_path"] = str(path)
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8", newline="\n")
    return str(path)


def _accelerator_cache_check(group: str, saved: dict, config: dict) -> dict:
    check = {
        "status": "not_requested",
        "mismatches": [],
        "accelerator": str(saved.get("accelerator") or ""),
        "verified": bool(saved.get("accelerator_verified", False)),
    }
    if not saved.get("accelerator_requested", False):
        return check
    decision = acceleration_install_decision(config, group)
    expected_accelerator = str(decision.get("accelerator") or "")
    saved_accelerator = str(saved.get("accelerator") or "")
    check["status"] = "validating"
    check["expected_accelerator"] = expected_accelerator
    if expected_accelerator != saved_accelerator:
        check["status"] = "stale"
        check["mismatches"].append("accelerator")
        return check
    if not saved.get("accelerator_verified", False):
        check["status"] = "stale"
        check["mismatches"].append("accelerator_unverified")
        return check
    if group == "onnx":
        from .dependency_manager import onnxruntime_available_providers

        provider = {
            "cuda": "CUDAExecutionProvider",
            "directml": "DmlExecutionProvider",
            "openvino": "OpenVINOExecutionProvider",
        }.get(saved_accelerator)
        providers, provider_error = onnxruntime_available_providers()
        check["providers"] = providers
        check["expected_provider"] = provider or ""
        if provider_error:
            check["status"] = "stale"
            check["mismatches"].append("provider_probe_error")
        elif provider and provider not in providers:
            check["status"] = "stale"
            check["mismatches"].append("accelerator_provider")
        else:
            check["status"] = "valid"
        return check
    if group == "faster_whisper" and saved_accelerator == "cuda":
        from .ctranslate2_probe import ctranslate2_cuda_available

        check["ctranslate2_cuda_available"] = ctranslate2_cuda_available()
        if check["ctranslate2_cuda_available"]:
            check["status"] = "valid"
        else:
            check["status"] = "stale"
            check["mismatches"].append("accelerator_backend")
        return check
    if group == "llama_cpp" and saved_accelerator in {"cuda", "vulkan"}:
        from .dependency_manager import llama_cpp_gpu_capable

        check["llama_cpp_gpu_capable"] = llama_cpp_gpu_capable()
        if check["llama_cpp_gpu_capable"]:
            check["status"] = "valid"
        else:
            check["status"] = "stale"
            check["mismatches"].append("accelerator_backend")
        return check
    check["status"] = "stale"
    check["mismatches"].append("accelerator_revalidation_unsupported")
    return check


def reusable_saved_runtime_resolution(group: str, config: dict, project_root: Path) -> dict:
    path = project_root / "Logs" / f"dependency_resolution_{group}.json"
    if not path.exists():
        return {
            "schema": "easy_asr_bench.runtime_resolution_reuse.v1",
            "status": "missing",
            "path": str(path),
            "mismatches": [],
        }
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "schema": "easy_asr_bench.runtime_resolution_reuse.v1",
            "status": "invalid_json",
            "path": str(path),
            "mismatches": ["json"],
            "error": str(exc),
        }
    current_config = {
        "provider": config.get("runtime", {}).get("provider", ""),
        "prefer_gpu": config.get("runtime", {}).get("prefer_gpu", ""),
        "fallback_to_cpu": config.get("runtime", {}).get("fallback_to_cpu", ""),
    }
    mismatches = []
    if saved.get("schema") != "easy_asr_bench.runtime_resolution.v1":
        mismatches.append("schema")
    if saved.get("dependency_group") != group:
        mismatches.append("dependency_group")
    if saved.get("config_runtime") != current_config:
        mismatches.append("config_runtime")
    if not saved.get("backend_verified", False):
        mismatches.append("backend_verified")
    accelerator_cache_check = _accelerator_cache_check(group, saved, config)
    mismatches.extend(accelerator_cache_check.get("mismatches", []))
    provider_mismatches = []
    if saved.get("providers"):
        if group == "onnx":
            from .dependency_manager import onnxruntime_available_providers

            current_providers, provider_error = onnxruntime_available_providers()
            for provider in saved.get("providers", []):
                if provider not in current_providers:
                    provider_mismatches.append(provider)
            if provider_error:
                mismatches.append("provider_probe_error")
        else:
            provider_mismatches = list(saved.get("providers", []))
    if provider_mismatches:
        mismatches.append("providers")
    runtime_path = str(saved.get("runtime_path") or "")
    runtime_path_missing = False
    if runtime_path:
        if group == "llama_mtmd":
            runtime_path_missing = not Path(runtime_path).exists()
        else:
            runtime_path_missing = True
    if runtime_path_missing:
        mismatches.append("runtime_path")
    missing = missing_modules_for_config(group, config)
    if missing:
        mismatches.append("missing")
    version_mismatches = []
    for package, version in (saved.get("versions") or {}).items():
        if not version:
            continue
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version_mismatches.append(package)
            continue
        if installed != version:
            version_mismatches.append(package)
    if version_mismatches:
        mismatches.append("versions")
    return {
        "schema": "easy_asr_bench.runtime_resolution_reuse.v1",
        "status": "valid" if not mismatches else "stale",
        "path": str(path),
        "mismatches": mismatches,
        "missing": missing,
        "version_mismatches": version_mismatches,
        "provider_mismatches": provider_mismatches,
        "runtime_path_missing": runtime_path_missing,
        "accelerator_cache_check": accelerator_cache_check,
        "resolution": saved,
    }


def build_repair_plan(config: dict, project_root: Path | None = None, status: dict[str, dict] | None = None) -> dict:
    project_root = project_root or Path(__file__).resolve().parent.parent
    status = status if status is not None else dependency_status(config)
    records = []
    for group, data in status.items():
        missing = list(data.get("missing") or [])
        acceleration = acceleration_install_decision(config, group)
        command = data.get("accelerator_recovery_command") if acceleration.get("use_accelerator") else data.get("recovery_command")
        can_auto_repair = bool(command) and bool(missing)
        repair_action = classify_repair_action(missing, can_auto_repair)
        records.append(
            {
                "issue_id": f"dependency_group:{group}",
                "status": "needs_repair" if missing else "ok",
                "severity": "dependency" if missing else "ok",
                "affected_dependency_group": group,
                "affected_models": [],
                "missing": missing,
                "can_auto_repair": can_auto_repair,
                "repair_action": repair_action,
                "repair_command": str(command or ""),
                "log_path": str(project_root / "Logs" / f"dependency_install_{group}.log") if can_auto_repair else "",
                "before": {
                    "available": bool(data.get("available")),
                    "install_kind": data.get("install_kind", "pip"),
                    "requirement_file": data.get("requirement_file", ""),
                    "accelerator": acceleration.get("accelerator"),
                    "use_accelerator": bool(acceleration.get("use_accelerator")),
                },
                "after": {},
                "block_reason": "" if can_auto_repair or not missing else "No repair command is available for this dependency group.",
            }
        )
    return {
        "schema": "easy_asr_bench.repair_plan.v1",
        "mode": "plan_only",
        "project_root": str(project_root),
        "records": records,
        "summary": {
            "total": len(records),
            "needs_repair": sum(1 for record in records if record["status"] == "needs_repair"),
            "can_auto_repair": sum(1 for record in records if record["can_auto_repair"]),
        },
    }


def classify_repair_action(missing: list[str], can_auto_repair: bool) -> str:
    if not missing:
        return "verify_cached"
    if not can_auto_repair:
        return "block_no_safe_repair"
    text = " ".join(str(item).lower() for item in missing)
    if any(marker in text for marker in ["outdated", "too old", "stale", "version outside", "wrong version", "upgrade"]):
        return "upgrade_outdated"
    if any(marker in text for marker in ["conflict", "conflicting", "cpu-only", "wrong provider", "provider hidden", "replace"]):
        return "replace_conflicting"
    return "install_missing"


def execute_repair_plan(config: dict, project_root: Path | None = None, status: dict[str, dict] | None = None) -> dict:
    project_root = project_root or Path(__file__).resolve().parent.parent
    plan = build_repair_plan(config, project_root=project_root, status=status)
    plan["mode"] = "repair_all_safe"
    repaired = 0
    failed = 0
    skipped = 0
    for record in plan["records"]:
        group = record["affected_dependency_group"]
        record["attempts"] = []
        if record["status"] != "needs_repair":
            if config.get("dependency_install", {}).get("use_cached_runtime_resolutions", True):
                cached = reusable_saved_runtime_resolution(group, config, project_root)
                record["cached_runtime_resolution_check"] = {key: value for key, value in cached.items() if key != "resolution"}
                if cached.get("status") == "valid":
                    record["after"] = {
                        "available": True,
                        "missing": [],
                        "repair_result": "already_ok_cached_resolution",
                        "repair_action": record.get("repair_action", "verify_cached"),
                        "backend_probe": {
                            "kind": "cached_runtime_resolution",
                            "ok": True,
                            "source_resolution_path": cached["path"],
                            "cached_backend_probe_kind": cached["resolution"].get("backend_probe_kind", ""),
                        },
                        "runtime_resolution_path": cached["path"],
                        "runtime_resolution": cached["resolution"],
                        "previous_runtime_resolution_check": {
                            "schema": "easy_asr_bench.runtime_resolution_check.v1",
                            "status": "valid",
                            "mismatches": [],
                            "checks": {"cached_resolution_reused": True},
                        },
                    }
                    continue
            backend_probe = backend_probe_for_group(group, config)
            record["after"] = {
                "available": True,
                "missing": [],
                "repair_result": "already_ok",
                "repair_action": record.get("repair_action", "verify_cached"),
                "backend_probe": backend_probe,
            }
            if backend_probe.get("ok", False):
                persisted = persist_runtime_resolution(record, config, project_root)
                record["after"]["runtime_resolution_path"] = persisted["path"]
                record["after"]["runtime_resolution"] = persisted["resolution"]
                record["after"]["previous_runtime_resolution_check"] = persisted["previous_resolution_check"]
            continue
        if not record["can_auto_repair"]:
            skipped += 1
            record["after"] = {
                "available": False,
                "missing": list(record["missing"]),
                "repair_result": "blocked",
                "repair_action": record.get("repair_action", "block_no_safe_repair"),
                "block_reason": record["block_reason"] or "No safe automatic repair is available.",
            }
            continue
        attempt = {
            "attempt_number": 1,
            "repair_action": record.get("repair_action", "install_missing"),
            "repair_command": record["repair_command"],
            "log_path": record["log_path"],
            "status": "running",
        }
        record["attempts"].append(attempt)
        try:
            decision = install_group_for_config(group, project_root, config, log_path=Path(record["log_path"]))
            after_missing = missing_modules_for_config(group, config)
            backend_probe = backend_probe_for_group(group, config)
        except Exception as exc:
            failed += 1
            attempt["status"] = "failed"
            attempt["error_type"] = type(exc).__name__
            attempt["message"] = str(exc) or type(exc).__name__
            record["status"] = "repair_failed"
            record["after"] = {
                "available": False,
                "missing": list(record["missing"]),
                "repair_result": "failed",
                "repair_action": record.get("repair_action", "install_missing"),
                "error_type": type(exc).__name__,
                "message": str(exc) or type(exc).__name__,
            }
            continue
        attempt["status"] = "pass"
        attempt["install_decision"] = decision or {}
        record["after"] = {
            "available": not bool(after_missing),
            "missing": list(after_missing),
            "repair_result": "repaired" if not after_missing else "still_missing",
            "repair_action": record.get("repair_action", "install_missing"),
            "backend_probe": backend_probe,
        }
        if not after_missing and backend_probe.get("ok", False):
            persisted = persist_runtime_resolution(record, config, project_root)
            record["after"]["runtime_resolution_path"] = persisted["path"]
            record["after"]["runtime_resolution"] = persisted["resolution"]
            record["after"]["previous_runtime_resolution_check"] = persisted["previous_resolution_check"]
        if after_missing:
            failed += 1
            record["status"] = "repair_failed"
        else:
            repaired += 1
            record["status"] = "repaired"
    plan["summary"].update(
        {
            "repaired": repaired,
            "failed": failed,
            "skipped": skipped,
            "attempted": sum(1 for record in plan["records"] if record.get("attempts")),
            "backend_probes": sum(1 for record in plan["records"] if record.get("after", {}).get("backend_probe")),
            "runtime_resolutions": sum(1 for record in plan["records"] if record.get("after", {}).get("runtime_resolution_path")),
            "cached_runtime_resolutions": sum(
                1 for record in plan["records"] if record.get("after", {}).get("repair_result") == "already_ok_cached_resolution"
            ),
            "previous_runtime_resolution_valid": sum(
                1 for record in plan["records"] if record.get("after", {}).get("previous_runtime_resolution_check", {}).get("status") == "valid"
            ),
            "previous_runtime_resolution_stale": sum(
                1 for record in plan["records"] if record.get("after", {}).get("previous_runtime_resolution_check", {}).get("status") == "stale"
            ),
            "backend_probe_failed": sum(
                1
                for record in plan["records"]
                if record.get("after", {}).get("backend_probe") and not record["after"]["backend_probe"].get("ok", False)
            ),
            "accelerator_probes": sum(
                1
                for record in plan["records"]
                if record.get("after", {}).get("backend_probe", {}).get("accelerator_probe", {}).get("requested", False)
            ),
            "accelerator_probe_failed": sum(
                1
                for record in plan["records"]
                if record.get("after", {}).get("backend_probe", {}).get("accelerator_probe", {}).get("requested", False)
                and not record["after"]["backend_probe"]["accelerator_probe"].get("ok", False)
            ),
        }
    )
    persist_repair_plan_evidence(plan, project_root)
    return plan
