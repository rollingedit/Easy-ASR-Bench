from __future__ import annotations

from pathlib import Path

from app.config import load_config
from app.media import prepare_audio
from qa.runtime_matrix.common import write_row
from qa.runtime_matrix.media_fixtures import create_media_fixture_set


def _config() -> dict:
    config = load_config(Path("config.json"))
    config["chunking"]["target_chunk_seconds"] = 30
    config["chunking"]["hard_max_chunk_seconds"] = 30
    return config


def _positive_media_row(row_id: str, evidence_dir: Path) -> dict:
    fixtures = create_media_fixture_set(evidence_dir / "media")
    return _validate_positive_media(row_id, evidence_dir, fixtures)


def _validate_positive_media(row_id: str, evidence_dir: Path, fixtures: dict[str, Path]) -> dict:
    config = _config()
    details: dict[str, dict] = {}
    artifacts = [fixtures["wav"], fixtures["mp3"], fixtures["mp4"]]
    for key in ["wav", "mp3", "mp4"]:
        wav_path, samples, chunks = prepare_audio(fixtures[key], evidence_dir / "normalized" / key, config)
        artifacts.append(wav_path)
        details[key] = {
            "source": str(fixtures[key]),
            "normalized_wav": str(wav_path),
            "sample_count": int(len(samples)),
            "chunk_count": len(chunks),
            "duration_seconds": float(len(samples)) / 16000.0,
        }
        if len(samples) == 0 or not chunks:
            return write_row(
                row_id,
                "fail",
                evidence_dir,
                summary=f"{key} media fixture normalized to empty audio.",
                details=details,
                artifacts=artifacts,
            )
    return write_row(
        row_id,
        "pass",
        evidence_dir,
        summary="Generated WAV, MP3, and MP4-with-audio fixtures all normalized and chunked successfully.",
        details=details,
        artifacts=artifacts,
    )


def _exercise_no_audio_fixture(evidence_dir: Path, fixtures: dict[str, Path]) -> tuple[bool, dict]:
    try:
        prepare_audio(fixtures["no_audio_mp4"], evidence_dir / "normalized" / "no_audio", _config())
    except Exception as exc:
        message = str(exc)
        lower = message.lower()
        readable_no_audio = "no audio stream" in lower or "does not contain any stream" in lower
        return readable_no_audio, {"error_type": type(exc).__name__, "message": message}
    return False, {"message": "Generated MP4-without-audio unexpectedly normalized successfully."}


def _no_audio_row(row_id: str, evidence_dir: Path) -> dict:
    fixtures = create_media_fixture_set(evidence_dir / "media")
    ok, details = _exercise_no_audio_fixture(evidence_dir, fixtures)
    return write_row(
        row_id,
        "pass" if ok else "fail",
        evidence_dir,
        summary="Generated MP4-without-audio produced a readable no-audio error." if ok else "Generated MP4-without-audio failed with the wrong error.",
        details=details,
        artifacts=[fixtures["no_audio_mp4"]],
    )


def _exercise_corrupt_fixture(evidence_dir: Path, fixtures: dict[str, Path]) -> tuple[bool, dict]:
    try:
        prepare_audio(fixtures["corrupt"], evidence_dir / "normalized" / "corrupt", _config())
    except Exception as exc:
        message = str(exc)
        readable = any(marker in message.lower() for marker in ["ffmpeg", "ffprobe", "inspect", "conversion", "invalid"])
        return readable, {"error_type": type(exc).__name__, "message": message}
    return False, {"message": "Generated corrupt media unexpectedly normalized successfully."}


def _corrupt_row(row_id: str, evidence_dir: Path) -> dict:
    fixtures = create_media_fixture_set(evidence_dir / "media")
    ok, details = _exercise_corrupt_fixture(evidence_dir, fixtures)
    return write_row(
        row_id,
        "pass" if ok else "fail",
        evidence_dir,
        summary="Generated corrupt media produced a readable media error." if ok else "Generated corrupt media failed without a readable media error.",
        details=details,
        artifacts=[fixtures["corrupt"]],
    )


def _combined_media_row(row_id: str, evidence_dir: Path) -> dict:
    fixtures = create_media_fixture_set(evidence_dir / "media")
    positive = _validate_positive_media(row_id, evidence_dir / "positive", fixtures)
    no_audio_ok, no_audio_details = _exercise_no_audio_fixture(evidence_dir, fixtures)
    corrupt_ok, corrupt_details = _exercise_corrupt_fixture(evidence_dir, fixtures)
    positive_ok = positive["status"] == "pass"
    details = {
        "audio_fixtures": positive["details"],
        "no_audio_video": no_audio_details,
        "corrupt_media": corrupt_details,
        "sub_row_status": {
            "wav_mp3_mp4_media": positive["status"],
            "no_audio_video_readable_error": "pass" if no_audio_ok else "fail",
            "corrupt_media_readable_error": "pass" if corrupt_ok else "fail",
        },
    }
    status = "pass" if positive_ok and no_audio_ok and corrupt_ok else "fail"
    return write_row(
        row_id,
        status,
        evidence_dir,
        summary=(
            "Generated WAV/MP3/MP4 audio fixtures normalized, MP4-without-audio produced a no-audio error, "
            "and corrupt media produced a readable media error."
            if status == "pass"
            else "One or more generated media fixture checks failed."
        ),
        details=details,
        artifacts=[fixtures["wav"], fixtures["mp3"], fixtures["mp4"], fixtures["no_audio_mp4"], fixtures["corrupt"]],
    )


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    if row_id == "wav_mp3_mp4_media":
        return _positive_media_row(row_id, evidence_dir)
    if row_id == "wav_mp3_mp4_no_audio_corrupt_media":
        return _combined_media_row(row_id, evidence_dir)
    if row_id == "no_audio_video_readable_error":
        return _no_audio_row(row_id, evidence_dir)
    if row_id == "corrupt_media_readable_error":
        return _corrupt_row(row_id, evidence_dir)
    return write_row(row_id, "fail", evidence_dir, summary=f"Unhandled media fixture row: {row_id}")
