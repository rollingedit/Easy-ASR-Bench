from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from .config import load_config, selected_variants
from .output_writer import write_report
from .utils import expand_inputs
from .benchmark import VariantMetrics


def check_config(config: dict) -> None:
    for key in ["advanced", "input", "chunking", "runtime", "transcription", "output"]:
        if key not in config:
            raise AssertionError(f"Missing config section {key}")
    for key in ["folders", "model_scan", "security"]:
        if key not in config:
            raise AssertionError(f"Missing product config section {key}")


def check_ffmpeg() -> None:
    from .media import ffmpeg_exe

    path = Path(ffmpeg_exe())
    if not path.exists():
        raise AssertionError(f"imageio-ffmpeg executable was not found: {path}")


def check_frontend_fixtures(config: dict) -> None:
    from .frontend import validate_against_fixture
    from .onnx_common import model_family_for_variant

    models_dir = Path(config["advanced"]["models_folder"])
    checked: set[Path] = set()
    for variant in selected_variants(config):
        root = models_dir / model_family_for_variant(variant)
        if root in checked or not root.exists():
            continue
        checked.add(root)
        result = validate_against_fixture(root)
        print(f"Frontend fixture {root}: {result}")


def check_queue_expansion(config: dict) -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        good = root / "sample.wav"
        bad = root / "notes.txt"
        good.write_bytes(b"")
        bad.write_text("not media", encoding="utf-8")
        files = expand_inputs([root], {".wav"}, True)
        if files != [good]:
            raise AssertionError("Queue expansion did not filter media extensions correctly")


def check_output_writer(config: dict) -> None:
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp)
        source = output / "sample.wav"
        source.write_bytes(b"")
        class Chunk:
            index = 0
            start_seconds = 0.0
            end_seconds = 1.0
        metrics = VariantMetrics("ar_int8", "int8", "CPUExecutionProvider", 1.0, 1)
        path = write_report(
            source,
            1.0,
            [Chunk()],
            {"ar_int8": {"chunks": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "hello"}], "metrics": metrics}},
            output,
        )
        text = path.read_text(encoding="utf-8")
        if "Granite Speech ONNX" not in text or "VARIANT: ar_int8" not in text:
            raise AssertionError("Output writer did not create expected sections")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--skip-model-validation", action="store_true")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    check_config(config)
    check_ffmpeg()
    check_frontend_fixtures(config)
    check_queue_expansion(config)
    check_output_writer(config)
    print("Self-test passed")


if __name__ == "__main__":
    main()
