from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from .adapters.base import ChunkTranscript, ModelCandidate, ModelRunResult
from .config import load_config
from .results_writer import build_results, write_all_reports
from .utils import expand_inputs


def check_config(config: dict) -> None:
    for key in ["folders", "input", "chunking", "runtime", "transcription", "reports", "model_scan", "security"]:
        if key not in config:
            raise AssertionError(f"Missing config section {key}")


def check_ffmpeg() -> None:
    try:
        from .media import ffmpeg_exe
    except ModuleNotFoundError as exc:
        raise AssertionError(f"Missing core dependency for media support: {exc.name}. Run setup.bat.") from exc

    path = Path(ffmpeg_exe())
    if not path.exists():
        raise AssertionError(f"imageio-ffmpeg executable was not found: {path}")


def check_queue_expansion() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        good = root / "sample.wav"
        bad = root / "notes.txt"
        good.write_bytes(b"")
        bad.write_text("not media", encoding="utf-8")
        files = expand_inputs([root], {".wav"}, True)
        if files != [good]:
            raise AssertionError("Queue expansion did not filter media extensions correctly")


def check_results_writer() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "sample.wav"
        source.write_bytes(b"fake")

        class Chunk:
            index = 0
            start_seconds = 0.0
            end_seconds = 1.0

        candidate = ModelCandidate(
            candidate_id="test_model",
            display_name="Test Model",
            family_name="test",
            backend="test",
            container_format="test",
            task="automatic-speech-recognition",
            precision="fp32",
            quantization_label="32-bit / FP32",
            path=root,
            adapter_name="test",
            runnable=True,
        )
        result = ModelRunResult(
            candidate,
            [ChunkTranscript("0001", 0.0, 1.0, "hello world")],
            {"audio_seconds": 1.0, "chunk_count": 1, "total_wall_seconds": 0.5, "peak_process_memory_mb": 1.0},
        )
        results = build_results(source, 1.0, [Chunk()], [result], [], 0.1)
        output_dir = write_all_reports(results, root / "Output")
        for name in ["results.txt", "results.json", "benchmark.csv", "compare.html"]:
            if not (output_dir / name).exists():
                raise AssertionError(f"Missing report output {name}")
        text = (output_dir / "results.txt").read_text(encoding="utf-8")
        if "LLM-Corrected Reference Instructions" not in text:
            raise AssertionError("TXT report is missing LLM reference instructions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    check_config(config)
    check_ffmpeg()
    check_queue_expansion()
    check_results_writer()
    print("Self-test passed")


if __name__ == "__main__":
    main()
