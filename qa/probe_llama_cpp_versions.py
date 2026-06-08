from __future__ import annotations

import ctypes
import json
from pathlib import Path
import subprocess
import sys


if sys.platform == "win32":
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)


VERSIONS = [
    "0.3.27",
    "0.3.26",
    "0.3.25",
    "0.3.24",
    "0.3.23",
    "0.3.22",
    "0.3.21",
    "0.3.20",
    "0.3.19",
    "0.3.18",
    "0.3.17",
    "0.3.16",
    "0.3.15",
    "0.3.14",
    "0.3.13",
    "0.3.12",
    "0.3.11",
    "0.3.10",
]


def main() -> int:
    model = Path(sys.argv[1])
    results = []
    probe = """
import ctypes
import json
import sys
if sys.platform == "win32":
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
from llama_cpp import Llama
llm = Llama(model_path=sys.argv[1], n_ctx=256, verbose=False)
out = llm("Say hi", max_tokens=4, temperature=0.0)
print(json.dumps({"text": out["choices"][0]["text"]}))
"""
    for version in VERSIONS:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-deps",
            "--extra-index-url",
            "https://abetlen.github.io/llama-cpp-python/whl/cpu",
            f"llama-cpp-python=={version}",
        ]
        print(f"INSTALL {version}", flush=True)
        installed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
        print((installed.stdout or "")[-1000:], flush=True)
        if installed.returncode != 0:
            results.append({"version": version, "install": "fail", "output": (installed.stdout or "")[-500:]})
            continue
        run = subprocess.run([sys.executable, "-c", probe, str(model)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        print(f"PROBE {version} {run.returncode}", flush=True)
        print((run.stdout or "")[-1000:], flush=True)
        results.append({"version": version, "returncode": run.returncode, "output": (run.stdout or "")[-500:]})
        if run.returncode == 0:
            print(f"FOUND {version}")
            print(json.dumps(results, indent=2))
            return 0
    print(json.dumps(results, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
