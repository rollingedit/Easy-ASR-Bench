import json
import re
from pathlib import Path

import numpy as np
import pytest

from app.html_report_builder import build_html_report
from app.model_scanner import scan_models
from app.results_writer import build_results


def test_release_gate_verifies_committed_metadata():
    workflow = Path(".github/workflows/release-gate.yml").read_text(encoding="utf-8")

    build_step = workflow.split("Build and validate release ZIP", 1)[1].split("Upload built release ZIP", 1)[0]
    assert "--update-metadata" not in build_step
    assert "release:" in workflow
    assert "types: [published, prereleased]" in workflow
    assert "release_tag" in workflow
    assert "github.event_name == 'release'" in workflow
    assert "actions/upload-artifact" in workflow
    assert "gh release download" in workflow
    assert "release-asset/checksums.json" in workflow


def test_release_builder_uses_working_tree_bytes_not_head_blobs():
    text = Path("scripts/build_release_zip.py").read_text(encoding="utf-8")

    assert "git show" not in text
    assert "(ROOT / rel).read_bytes()" in text
    assert '!= "installer/checksums.json"' in text
    assert "--strict-checksums" in text


def test_installer_uses_tls12_and_basic_parsing_for_downloads():
    text = Path("installer/install.ps1").read_text(encoding="utf-8")

    assert "SecurityProtocolType]::Tls12" in text
    for line in text.splitlines():
        if "Invoke-WebRequest" in line:
            assert "-UseBasicParsing" in line


def test_readme_keeps_openai_pt_out_of_default_runnable_models():
    readme = Path("README.md").read_text(encoding="utf-8")
    runnable_section = readme.split("### Runnable ASR Models", 1)[1].split("### Blocked By Default", 1)[0]

    assert ".pt" not in runnable_section
    assert "OpenAI Whisper `.pt` checkpoint files are detected, but blocked by default" in readme


def test_html_scoring_uses_unicode_normalization_and_escapes_metadata():
    html = build_html_report(
        {
            "source": {"name": "x.wav", "duration_seconds": 1, "sha256": "abc"},
            "chunk_plan": {"chunks": [{"chunk_id": "0001", "start_timestamp": "00:00:00.000", "end_timestamp": "00:00:01.000"}]},
            "runs": [
                {
                    "model": {
                        "candidate_id": "bad",
                        "display_name": "<img src=x onerror=alert(1)>",
                        "precision": "fp32",
                        "backend": "test",
                    },
                    "transcript_chunks": [{"chunk_id": "0001", "text": "Cafe Beijing don't"}],
                    "metrics": {},
                    "errors": [],
                }
            ],
            "pairwise_differences": {},
        }
    )

    assert "\\p{L}" in html
    assert "normalize('NFKC')" in html
    assert "source_sha256" in html
    assert "Duplicate chunks" in html
    assert "${safe(run.model.display_name)}" in html
    assert "${run.model.display_name}" not in html


def test_hf_whisper_folder_is_not_discovered_twice(tmp_path: Path):
    model = tmp_path / "whisper"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "whisper"}), encoding="utf-8")
    (model / "model.safetensors").write_text("", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    assert [candidate.adapter_name for candidate in runnable] == ["hf_whisper_asr"]
    assert unsupported == []


def test_incomplete_granite_folder_is_reported_once(tmp_path: Path):
    model = tmp_path / "granite"
    precision = model / "fp32"
    precision.mkdir(parents=True)
    (precision / "encoder.onnx").write_text("", encoding="utf-8")

    runnable, unsupported = scan_models(tmp_path)

    assert runnable == []
    granite_entries = [candidate for candidate in unsupported if "granite" in candidate.candidate_id]
    assert len(granite_entries) == 1


def test_chunk_plan_records_real_cut_reason_and_rms():
    pytest.importorskip("imageio_ffmpeg")
    from app.media import plan_chunks

    sr = 16000
    tone = np.full(sr, 0.2, dtype=np.float32)
    silence = np.zeros(sr // 2, dtype=np.float32)
    samples = np.concatenate([tone, silence, tone])
    config = {
        "chunking": {
            "target_chunk_seconds": 1.0,
            "hard_max_chunk_seconds": 1.2,
            "boundary_search_seconds": 0.4,
            "silence_threshold_db": -35,
            "rms_fallback_window_ms": 100,
            "allow_overlap": False,
        }
    }

    chunks = plan_chunks(samples, config, sr)

    assert len(chunks) >= 2
    assert chunks[0].cut_reason == "silence"
    assert chunks[0].rms_db <= -35


def test_results_environment_and_chunk_metadata_are_not_stubs(tmp_path: Path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    chunk = type(
        "Chunk",
        (),
        {"index": 0, "start_seconds": 0.0, "end_seconds": 1.0, "cut_reason": "end_of_audio", "rms_db": -42.0},
    )()

    results = build_results(source, 1.0, [chunk], [], [], 0.1)

    assert results["environment"]["python"] != "local"
    assert results["environment"]["platform"] != "windows" or results["environment"]["system"] == "Windows"
    assert "dependency_versions" in results
    assert "adapter_versions" in results
    assert results["chunk_plan"]["chunks"][0]["cut_reason"] == "end_of_audio"
    assert results["chunk_plan"]["chunks"][0]["rms_db"] == -42.0
