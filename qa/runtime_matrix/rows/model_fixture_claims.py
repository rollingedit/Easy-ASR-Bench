from __future__ import annotations

import json
from pathlib import Path

from qa.runtime_matrix.common import ROOT, write_row


MANIFEST = ROOT / "qa" / "runtime_matrix" / "model_fixtures.json"


def _quality_rows(rows: list[str]) -> list[str]:
    quality_markers = ("quality", "speech", "real_tiny", "audio_asr_gguf_mmproj", "public_download_to_asr")
    return [row for row in rows if any(marker in row for marker in quality_markers)]


def _manifest_claim_failures(fixtures: dict) -> list[str]:
    failures: list[str] = []
    for fixture_id, fixture in fixtures.items():
        kind = str(fixture.get("kind", ""))
        rows = list(fixture.get("rows", []))
        notes = str(fixture.get("notes", "")).lower()
        structural = "structural" in kind or "generated" in kind or "random" in fixture_id
        not_quality_note = "not quality-bearing" in notes or "not a quality-bearing" in notes or "not be used for transcript-quality claims" in notes
        quality = "quality" in kind or ("quality-bearing" in notes and not not_quality_note)
        if structural:
            if not not_quality_note:
                failures.append(f"{fixture_id} is structural but lacks an explicit not-quality-bearing note")
            if "quality-bearing" in notes and not not_quality_note:
                failures.append(f"{fixture_id} structural note contains ambiguous quality-bearing language")
        if quality:
            if not _quality_rows(rows):
                failures.append(f"{fixture_id} claims quality coverage but has no quality/WER runtime row")
            if not any(marker in notes for marker in ["wer", "transcript", "quality-bearing"]):
                failures.append(f"{fixture_id} quality claim lacks WER/transcript wording")
        if fixture_id.startswith("same_media_multi_model"):
            included = set(fixture.get("includes", []))
            structural_includes = [
                included_id
                for included_id in included
                if "structural" in str(fixtures.get(included_id, {}).get("kind", "")) or "random" in included_id
            ]
            if structural_includes and "structural" not in notes:
                failures.append(f"{fixture_id} includes structural fixtures without disclosing them")
            if structural_includes and "not quality-bearing" not in notes and "mixed-provider" not in notes:
                failures.append(f"{fixture_id} does not separate structural includes from quality claims")
    return failures


def run(row_id: str, evidence_dir: Path, _install_deps: bool, _allow_downloads: bool) -> dict:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fixtures = data.get("fixtures", {})
    failures = []
    if data.get("schema") != "easy_asr_bench.runtime_matrix.model_fixtures.v1":
        failures.append("model fixture manifest schema marker is missing")
    failures.extend(_manifest_claim_failures(fixtures))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_copy = evidence_dir / "model_fixtures_claims_checked.json"
    manifest_copy.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    structural_count = sum(1 for fixture_id, fixture in fixtures.items() if "structural" in str(fixture.get("kind", "")) or "random" in fixture_id)
    quality_count = sum(1 for fixture in fixtures.values() if "quality" in str(fixture.get("kind", "")) or "quality-bearing" in str(fixture.get("notes", "")).lower())
    return write_row(
        row_id,
        "pass" if not failures else "fail",
        evidence_dir,
        summary=(
            "Model fixture manifest separates structural fixtures from quality-bearing WER/transcript claims."
            if not failures
            else "Model fixture quality-claim validation failed."
        ),
        details={
            "fixture_count": len(fixtures),
            "structural_fixture_count": structural_count,
            "quality_fixture_count": quality_count,
            "failures": failures,
        },
        artifacts=[manifest_copy],
    )
