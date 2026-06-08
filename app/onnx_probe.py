from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def _windows_error_mode_prefix() -> str:
    return (
        "import sys\n"
        "if sys.platform == 'win32':\n"
        "    import ctypes\n"
        "    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)\n"
    )


def probe_onnx_sessions(model_paths: list[Path], providers: list[str], cpu_threads: int = 0, timeout: int = 30) -> str:
    payload = {
        "model_paths": [str(path) for path in model_paths],
        "providers": list(providers),
        "cpu_threads": int(cpu_threads),
    }
    code = (
        _windows_error_mode_prefix()
        + """
import json
import sys

payload = json.loads(sys.argv[1])
try:
    import onnxruntime as ort
    options = ort.SessionOptions()
    cpu_threads = int(payload.get("cpu_threads") or 0)
    if cpu_threads:
        options.intra_op_num_threads = cpu_threads
        options.inter_op_num_threads = cpu_threads
    providers = list(payload.get("providers") or ["CPUExecutionProvider"])
    if "DmlExecutionProvider" in providers:
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    loaded = []
    for model_path in payload["model_paths"]:
        session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
        loaded.append({"model": model_path, "providers": list(session.get_providers())})
    print(json.dumps({"ok": True, "loaded": loaded}))
except BaseException as exc:
    print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}))
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
        return f"{type(exc).__name__}: {exc}"
    stdout = completed.stdout.strip()
    if completed.returncode == 0:
        return ""
    try:
        result = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        detail = (completed.stderr or stdout or f"exit {completed.returncode}").strip()
        return detail
    return f"{result.get('error_type', 'Error')}: {result.get('error', '')}"
