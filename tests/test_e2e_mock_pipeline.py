import json
from pathlib import Path

import numpy as np

from app.adapters.base import ChunkTranscript, ModelCandidate, ModelRunResult
from app.main import output_status, process_file_with_candidates
from app.media import AudioChunk


class PassingAdapter:
    name = "fake_pass"

    def __init__(self):
        self.loaded = False

    def required_dependency_groups(self, candidate):
        return []

    def load(self, candidate, runtime_config):
        self.loaded = True

    def transcribe_chunks(self, chunks, chunk_metadata):
        candidate = self.candidate
        return ModelRunResult(
            candidate=candidate,
            transcript_chunks=[
                ChunkTranscript(
                    chunk_id=meta["chunk_id"],
                    start_seconds=meta["start_seconds"],
                    end_seconds=meta["end_seconds"],
                    text=f"transcript {meta['chunk_id']}",
                )
                for meta in chunk_metadata
            ],
            metrics={"inference_seconds": 0.01, "audio_seconds": 1.0, "peak_process_memory_mb": 123},
        )

    def unload(self):
        self.loaded = False


class ChunkFailingAdapter(PassingAdapter):
    name = "fake_chunk_fail"

    def transcribe_chunks(self, chunks, chunk_metadata):
        candidate = self.candidate
        return ModelRunResult(
            candidate=candidate,
            transcript_chunks=[],
            metrics={"inference_seconds": 0.01, "audio_seconds": 1.0, "peak_process_memory_mb": 123},
            errors=["0001: CUDA backend failed for chunk"],
        )


class FailingAdapter(PassingAdapter):
    name = "fake_fail"

    def load(self, candidate, runtime_config):
        raise RuntimeError("model load failed")


def candidate(candidate_id: str, adapter_name: str) -> ModelCandidate:
    return ModelCandidate(
        candidate_id=candidate_id,
        display_name=candidate_id,
        family_name="Fixture",
        backend="fixture",
        container_format="fixture",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="32-bit / FP32",
        path=Path(candidate_id),
        adapter_name=adapter_name,
        runnable=True,
    )


def test_mock_e2e_pipeline_writes_reports_when_one_model_fails(tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    source.write_bytes(b"fixture")
    temp = tmp_path / "Temp"
    output = tmp_path / "Output"
    chunk = AudioChunk(0, 0.0, 1.0, np.zeros(16000, dtype=np.float32))
    wav_path = temp / "normalized.wav"
    wav_path.parent.mkdir()
    wav_path.write_bytes(b"wav")

    monkeypatch.setattr("app.main.wait_for_stable_file", lambda path, seconds: None)
    monkeypatch.setattr("app.media.prepare_audio", lambda source, temp_dir, config: (wav_path, np.zeros(16000, dtype=np.float32), [chunk]))
    monkeypatch.setattr("app.media.audio_duration_seconds", lambda samples: 1.0)

    def fake_adapter_for(model):
        if model.adapter_name == "fake_pass":
            adapter = PassingAdapter()
        elif model.adapter_name == "fake_chunk_fail":
            adapter = ChunkFailingAdapter()
        else:
            adapter = FailingAdapter()
        adapter.candidate = model
        return adapter

    monkeypatch.setattr("app.main.adapter_for", fake_adapter_for)
    config = {
        "folders": {"temp": str(temp), "output": str(output)},
        "input": {"file_stability_wait_seconds": 0},
        "runtime": {"provider": "auto"},
        "advanced": {"keep_temp_wavs": False},
    }

    report_dir = process_file_with_candidates(
        source,
        [candidate("good-model", "fake_pass"), candidate("bad-model", "fake_fail")],
        config,
    )

    assert report_dir is not None
    for name in ["results.json", "results.txt", "benchmark.csv", "compare.html", "results_llm_prompt_part_001.txt"]:
        assert (report_dir / name).exists()
    results = json.loads((report_dir / "results.json").read_text(encoding="utf-8"))
    assert [run["model"]["candidate_id"] for run in results["runs"]] == ["good-model", "bad-model"]
    good, bad = results["runs"]
    assert good["transcript_chunks"][0]["text"] == "transcript 0001"
    assert bad["transcript_chunks"] == []
    assert bad["errors"][0]["status"] == "model_failed"
    assert bad["errors"][0]["stage"] == "model_load_or_inference"
    assert bad["errors"][0]["message"] == "model load failed"
    assert bad["errors"][0]["model_id"] == "bad-model"
    assert bad["errors"][0]["likely_causes"]
    assert bad["errors"][0]["next_actions"]
    assert bad["errors"][0]["traceback"]
    assert "Likely causes" in (report_dir / "results.txt").read_text(encoding="utf-8")
    assert "Model Errors" in (report_dir / "compare.html").read_text(encoding="utf-8")
    assert not wav_path.exists()


def test_mock_e2e_pipeline_structures_chunk_errors(tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    source.write_bytes(b"fixture")
    temp = tmp_path / "Temp"
    output = tmp_path / "Output"
    chunk = AudioChunk(0, 0.0, 1.0, np.zeros(16000, dtype=np.float32))
    wav_path = temp / "normalized.wav"
    wav_path.parent.mkdir()
    wav_path.write_bytes(b"wav")

    monkeypatch.setattr("app.main.wait_for_stable_file", lambda path, seconds: None)
    monkeypatch.setattr("app.media.prepare_audio", lambda source, temp_dir, config: (wav_path, np.zeros(16000, dtype=np.float32), [chunk]))
    monkeypatch.setattr("app.media.audio_duration_seconds", lambda samples: 1.0)

    def fake_adapter_for(model):
        adapter = ChunkFailingAdapter()
        adapter.candidate = model
        return adapter

    monkeypatch.setattr("app.main.adapter_for", fake_adapter_for)

    report_dir = process_file_with_candidates(
        source,
        [candidate("chunk-fail-model", "fake_chunk_fail")],
        {
            "folders": {"temp": str(temp), "output": str(output)},
            "input": {"file_stability_wait_seconds": 0},
            "runtime": {"provider": "auto"},
            "advanced": {"keep_temp_wavs": False},
        },
    )

    results = json.loads((report_dir / "results.json").read_text(encoding="utf-8"))
    error = results["runs"][0]["errors"][0]

    assert error["status"] == "chunk_failed"
    assert error["stage"] == "chunk_inference"
    assert error["chunk_id"] == "0001"
    assert error["likely_causes"]
    assert error["next_actions"]
    assert "CUDA backend failed" in error["message"]


def test_mock_e2e_pipeline_writes_failed_file_report_when_media_preparation_fails(tmp_path, monkeypatch, capsys):
    source = tmp_path / "broken.mp4"
    source.write_bytes(b"fixture")
    monkeypatch.setattr("app.main.wait_for_stable_file", lambda path, seconds: None)
    monkeypatch.setattr("app.media.prepare_audio", lambda source, temp_dir, config: (_ for _ in ()).throw(RuntimeError("no audio stream")))

    report_dir = process_file_with_candidates(
        source,
        [candidate("good-model", "fake_pass")],
        {
            "folders": {"temp": str(tmp_path / "Temp"), "output": str(tmp_path / "Output")},
            "input": {"file_stability_wait_seconds": 0},
            "runtime": {"provider": "auto"},
            "advanced": {"keep_temp_wavs": False},
        },
    )

    assert report_dir is not None
    for name in ["results.json", "results.txt", "benchmark.csv", "compare.html"]:
        assert (report_dir / name).exists()
    results = json.loads((report_dir / "results.json").read_text(encoding="utf-8"))
    assert results["runs"] == []
    assert results["errors"][0]["status"] == "failed_before_model_run"
    assert results["errors"][0]["stage"] == "media_probe"
    assert "no audio stream" in results["errors"][0]["message"]
    assert output_status(report_dir) == "failed"
    assert "No source files were modified" in (report_dir / "results.txt").read_text(encoding="utf-8")
    html = (report_dir / "compare.html").read_text(encoding="utf-8")
    assert "Run Status" in html
    assert "No source files were modified" in html
    assert "Wrote failure report" in capsys.readouterr().out
