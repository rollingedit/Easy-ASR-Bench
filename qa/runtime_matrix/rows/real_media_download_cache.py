from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from app.config import load_config
from app.media import prepare_audio
from qa.runtime_matrix.common import ROOT, sha256, write_row


MANIFEST = ROOT / "qa" / "runtime_matrix" / "real_media_fixtures.json"


def _config() -> dict:
    config = load_config(Path("config.json"))
    config["chunking"]["target_chunk_seconds"] = 30
    config["chunking"]["hard_max_chunk_seconds"] = 30
    return config


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Easy-ASR-Bench-runtime-matrix"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _extension_for_fixture(name: str, fixture: dict) -> str:
    kind = str(fixture.get("kind", ""))
    if kind.endswith("_wav"):
        return ".wav"
    if kind.endswith("_ogg"):
        return ".ogg"
    if "webm" in kind:
        return ".webm"
    if "mp4" in kind:
        return ".mp4"
    return Path(name).suffix or ".bin"


def run(row_id: str, evidence_dir: Path, _install_deps: bool, allow_downloads: bool) -> dict:
    manifest = _load_manifest()
    fixtures = manifest.get("fixtures", {})
    downloadable = {name: fixture for name, fixture in fixtures.items() if fixture.get("download_url")}
    pending_source_only = {
        name: {
            "kind": fixture.get("kind"),
            "source_page": fixture.get("source_page"),
            "license": fixture.get("license"),
            "reason": "manifest fixture has no stable download_url yet",
        }
        for name, fixture in fixtures.items()
        if not fixture.get("download_url")
    }
    details: dict[str, object] = {
        "manifest": str(MANIFEST),
        "downloadable_fixture_count": len(downloadable),
        "source_only_fixtures": pending_source_only,
    }
    if not allow_downloads:
        return write_row(
            row_id,
            "blocked",
            evidence_dir,
            summary="Real media fixture downloads require --allow-downloads.",
            details=details,
            block_reason="network downloads are disabled for this row",
            external_requirement="rerun with --allow-downloads after source/license review",
        )

    artifacts: list[Path] = []
    downloaded: dict[str, dict] = {}
    for name, fixture in downloadable.items():
        target = evidence_dir / "real_media" / f"{name}{_extension_for_fixture(name, fixture)}"
        try:
            _download(str(fixture["download_url"]), target)
        except Exception as exc:
            return write_row(
                row_id,
                "fail",
                evidence_dir,
                summary=f"Could not download real media fixture {name}.",
                details={**details, "failed_fixture": name, "error_type": type(exc).__name__, "message": str(exc)},
                artifacts=artifacts,
            )
        artifacts.append(target)
        digest = sha256(target)
        expected_prefix = fixture.get("expected_sha256_prefix")
        if expected_prefix and not digest.removeprefix("sha256:").startswith(str(expected_prefix)):
            return write_row(
                row_id,
                "fail",
                evidence_dir,
                summary=f"Downloaded real media fixture {name} did not match its expected SHA256 prefix.",
                details={**details, "failed_fixture": name, "expected_sha256_prefix": expected_prefix, "actual_sha256": digest},
                artifacts=artifacts,
            )
        downloaded[name] = {
            "kind": fixture.get("kind"),
            "source_page": fixture.get("source_page"),
            "download_url": fixture.get("download_url"),
            "license": fixture.get("license"),
            "sha256": digest,
            "bytes": target.stat().st_size,
        }
        if str(fixture.get("kind", "")).startswith("real_audio_"):
            wav_path, samples, chunks = prepare_audio(target, evidence_dir / "normalized" / name, _config())
            artifacts.append(wav_path)
            downloaded[name]["normalized_wav"] = str(wav_path)
            downloaded[name]["sample_count"] = int(len(samples))
            downloaded[name]["chunk_count"] = len(chunks)
            if len(samples) == 0 or not chunks:
                return write_row(
                    row_id,
                    "fail",
                    evidence_dir,
                    summary=f"Downloaded real audio fixture {name} normalized to empty audio.",
                    details={**details, "downloaded": downloaded},
                    artifacts=artifacts,
                )
        kind = str(fixture.get("kind", ""))
        if kind in {"real_video_mp4_with_audio", "real_video_webm_with_audio"}:
            wav_path, samples, chunks = prepare_audio(target, evidence_dir / "normalized" / name, _config())
            artifacts.append(wav_path)
            downloaded[name]["normalized_wav"] = str(wav_path)
            downloaded[name]["sample_count"] = int(len(samples))
            downloaded[name]["chunk_count"] = len(chunks)
            if len(samples) == 0 or not chunks:
                return write_row(
                    row_id,
                    "fail",
                    evidence_dir,
                    summary=f"Downloaded real media fixture {name} normalized to empty audio.",
                    details={**details, "downloaded": downloaded},
                    artifacts=artifacts,
                )
        if kind.startswith("real_video_") and kind.endswith("_no_audio"):
            try:
                prepare_audio(target, evidence_dir / "normalized" / name, _config())
            except Exception as exc:
                downloaded[name]["no_audio_error"] = str(exc)
                if "no audio stream" not in str(exc).lower():
                    return write_row(
                        row_id,
                        "fail",
                        evidence_dir,
                        summary=f"Downloaded no-audio media fixture {name} did not produce a readable no-audio error.",
                        details={**details, "downloaded": downloaded},
                        artifacts=artifacts,
                    )
            else:
                return write_row(
                    row_id,
                    "fail",
                    evidence_dir,
                    summary=f"Downloaded no-audio media fixture {name} unexpectedly normalized successfully.",
                    details={**details, "downloaded": downloaded},
                    artifacts=artifacts,
                )

    return write_row(
        row_id,
        "pass",
        evidence_dir,
        summary="Downloaded and cached real media fixtures with stable URLs, recorded hashes, and validated real audio plus video audio/no-audio behavior.",
        details={**details, "downloaded": downloaded},
        artifacts=artifacts,
    )
