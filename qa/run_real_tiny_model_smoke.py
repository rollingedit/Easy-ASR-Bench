from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.dependency_manager import install_group_for_config, missing_modules_for_config
from app.hf_model_downloader import RECOMMENDED_BASELINE_REPO, download_hf_model_from_ref
from app.main import process_file_with_candidates
from app.model_scanner import scan_models
from app.scoring import wer


REFERENCE_TEXT = "easy asr bench real model smoke test"


def suppress_windows_native_crash_dialogs() -> None:
    if sys.platform != "win32":
        return
    try:
        sem_failcriticalerrors = 0x0001
        sem_nogpfaulterrorbox = 0x0002
        sem_noopenfileerrorbox = 0x8000
        ctypes.windll.kernel32.SetErrorMode(sem_failcriticalerrors | sem_nogpfaulterrorbox | sem_noopenfileerrorbox)
    except Exception:
        return


def generate_windows_sapi_wav(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{str(path).replace("'", "''")}'); "
            f"$s.Speak('{text.replace("'", "''")}'); "
            "$s.Dispose()"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if completed.returncode != 0 or not path.exists() or path.stat().st_size < 1000:
        detail = (completed.stderr or completed.stdout or "no speech synthesis detail").strip()
        raise RuntimeError(f"Could not generate Windows SAPI speech WAV: {detail}")


def smoke_config(workdir: Path, provider: str) -> dict:
    config = load_config(ROOT / "config.json")
    config["folders"] = {
        "models": str(workdir / "Models"),
        "input": str(workdir / "Input"),
        "output": str(workdir / "Output"),
        "temp": str(workdir / "Temp"),
        "logs": str(workdir / "Logs"),
        "cache": str(workdir / "Cache"),
    }
    config["runtime"] = dict(config["runtime"])
    config["runtime"]["provider"] = provider
    config["runtime"]["prefer_gpu"] = provider != "cpu"
    config["runtime"]["fallback_to_cpu"] = True
    config["input"] = dict(config["input"])
    config["input"]["file_stability_wait_seconds"] = 0
    config["advanced"] = dict(config["advanced"])
    config["advanced"]["keep_temp_wavs"] = False
    return config


def ensure_faster_whisper_model(models_root: Path, model_ref: str) -> Path:
    runnable, _ = scan_models(models_root)
    candidates = [candidate for candidate in runnable if candidate.adapter_name == "faster_whisper"]
    if candidates:
        return candidates[0].path
    destination = download_hf_model_from_ref(models_root, model_ref, input_func=lambda _prompt="": "1")
    if destination is None:
        raise RuntimeError(f"Could not download real smoke model: {model_ref}")
    return destination


def assert_smoke_report(report_dir: Path, reference_text: str, max_normalized_wer: float) -> dict:
    required = ["results.json", "results.txt", "benchmark.csv", "compare.html"]
    missing = [name for name in required if not (report_dir / name).exists()]
    if missing:
        raise AssertionError("real smoke report missing artifacts: " + ", ".join(missing))
    results = json.loads((report_dir / "results.json").read_text(encoding="utf-8"))
    runs = results.get("runs") or []
    if not runs:
        raise AssertionError("real smoke produced no model runs")
    transcript = "\n".join(chunk.get("text", "") for chunk in runs[0].get("transcript_chunks", []))
    if not transcript.strip():
        raise AssertionError("real smoke transcript was empty")
    normalized_wer = wer(reference_text, transcript, normalized=True)
    if normalized_wer > max_normalized_wer:
        raise AssertionError(
            f"real smoke normalized WER {normalized_wer:.3f} exceeded threshold {max_normalized_wer:.3f}. "
            f"reference={reference_text!r}; transcript={transcript!r}"
        )
    runs[0].setdefault("metrics", {})["real_smoke_reference_text"] = reference_text
    runs[0]["metrics"]["real_smoke_normalized_wer"] = normalized_wer
    runs[0]["metrics"]["real_smoke_max_normalized_wer"] = max_normalized_wer
    metrics = runs[0].get("metrics", {})
    if "vram_measurement_source" not in metrics:
        raise AssertionError("real smoke metrics did not include vram_measurement_source")
    return results


def main() -> None:
    suppress_windows_native_crash_dialogs()
    parser = argparse.ArgumentParser(description="Run a real tiny ASR model smoke test through the normal report pipeline.")
    parser.add_argument("--workdir", default="Temp/real_tiny_model_smoke")
    parser.add_argument("--model-ref", default=RECOMMENDED_BASELINE_REPO)
    parser.add_argument("--provider", default="cpu", choices=["cpu", "auto", "cuda"])
    parser.add_argument("--reference-text", default=REFERENCE_TEXT)
    parser.add_argument("--max-normalized-wer", type=float, default=0.60)
    parser.add_argument("--install-deps", action="store_true", help="Install the faster-whisper dependency group if it is missing.")
    parser.add_argument("--clean", action="store_true", help="Remove the smoke workdir before running.")
    args = parser.parse_args()

    workdir = (ROOT / args.workdir).resolve()
    if args.clean and workdir.exists():
        shutil.rmtree(workdir)
    config = smoke_config(workdir, args.provider)
    for folder in config["folders"].values():
        Path(folder).mkdir(parents=True, exist_ok=True)

    missing = missing_modules_for_config("faster_whisper", config)
    if missing:
        if not args.install_deps:
            raise SystemExit(
                "Missing faster-whisper runtime modules: "
                + ", ".join(missing)
                + ". Re-run with --install-deps or install via setup.bat."
            )
        install_group_for_config("faster_whisper", ROOT, config, log_path=Path(config["folders"]["logs"]) / "real_tiny_model_smoke_dependency_install.log")
        missing = missing_modules_for_config("faster_whisper", config)
        if missing:
            raise SystemExit("faster-whisper runtime is still missing after install: " + ", ".join(missing))

    ensure_faster_whisper_model(Path(config["folders"]["models"]), args.model_ref)
    runnable, unsupported = scan_models(Path(config["folders"]["models"]))
    candidates = [candidate for candidate in runnable if candidate.adapter_name == "faster_whisper"]
    if not candidates:
        raise SystemExit("No runnable faster-whisper model was found after download.")

    source = Path(config["folders"]["input"]) / "real_tiny_model_smoke.wav"
    generate_windows_sapi_wav(source, args.reference_text)
    report_dir = process_file_with_candidates(source, [candidates[0]], config, unsupported)
    if report_dir is None:
        raise SystemExit("Real tiny model smoke did not produce a report directory.")
    results = assert_smoke_report(report_dir, args.reference_text, args.max_normalized_wer)
    payload = {
        "schema": "easy_asr_bench.real_tiny_model_smoke.v1",
        "status": "pass",
        "model": results["runs"][0]["model"],
        "metrics": results["runs"][0]["metrics"],
        "report_dir": str(report_dir),
        "source": str(source),
    }
    (report_dir / "real_tiny_model_smoke.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
