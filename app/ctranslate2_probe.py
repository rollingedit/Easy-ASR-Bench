from __future__ import annotations

import json
import subprocess
import sys


def _windows_error_mode_prefix() -> str:
    return (
        "import sys\n"
        "if sys.platform == 'win32':\n"
        "    import ctypes\n"
        "    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)\n"
    )


def ctranslate2_cuda_available() -> bool:
    result = ctranslate2_probe()
    return bool(result.get("cuda_available", False))


def ctranslate2_probe() -> dict:
    code = (
        _windows_error_mode_prefix()
        + """
import json
try:
    import ctranslate2
    cuda_available = False
    if hasattr(ctranslate2, "get_cuda_device_count"):
        cuda_available = int(ctranslate2.get_cuda_device_count()) > 0
    elif hasattr(ctranslate2, "get_supported_compute_types"):
        cuda_available = bool(ctranslate2.get_supported_compute_types("cuda"))
    print(json.dumps({"ok": True, "cuda_available": cuda_available, "version": getattr(ctranslate2, "__version__", "")}))
except Exception as exc:
    print(json.dumps({"ok": False, "cuda_available": False, "error": str(exc), "error_type": type(exc).__name__}))
"""
    )
    try:
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "cuda_available": False, "error": str(exc), "error_type": type(exc).__name__}
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
        return {"ok": False, "cuda_available": False, "error": detail, "error_type": "NativeProbeFailed", "returncode": completed.returncode}
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        return {"ok": False, "cuda_available": False, "error": str(exc), "error_type": type(exc).__name__}
