from argparse import Namespace
from pathlib import Path

from app.adapters.base import ModelCandidate
from app.batch_resume import BatchResumeManifest, batch_signature


def candidate(candidate_id: str, tmp_path: Path) -> ModelCandidate:
    return ModelCandidate(
        candidate_id=candidate_id,
        display_name=candidate_id,
        family_name="Test",
        backend="fixture",
        container_format="fixture",
        task="automatic-speech-recognition",
        precision="fp32",
        quantization_label="32-bit / FP32",
        path=tmp_path / candidate_id,
        adapter_name="fixture_asr",
        runnable=True,
        category="asr",
    )


def test_batch_resume_manifest_records_and_resolves_completed_pairs(tmp_path: Path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    output = tmp_path / "Output" / "run"
    output.mkdir(parents=True)
    (output / "results.json").write_text("{}", encoding="utf-8")
    selected = [candidate("one", tmp_path), candidate("two", tmp_path)]
    signature = batch_signature({"runtime": {"provider": "cpu"}}, selected)
    manifest = BatchResumeManifest(tmp_path / "Logs" / "batch_resume_manifest.json")

    manifest.record_file(source, selected, signature, output, "done")

    assert manifest.completed_output_for(source, selected, signature) == str(output)
    saved = BatchResumeManifest(tmp_path / "Logs" / "batch_resume_manifest.json")
    assert saved.completed_output_for(source, selected, signature) == str(output)


def test_batch_resume_manifest_invalidates_stale_input_or_config(tmp_path: Path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    output = tmp_path / "Output" / "run"
    output.mkdir(parents=True)
    (output / "results.json").write_text("{}", encoding="utf-8")
    selected = [candidate("one", tmp_path)]
    signature = batch_signature({"runtime": {"provider": "cpu"}}, selected)
    manifest = BatchResumeManifest(tmp_path / "Logs" / "batch_resume_manifest.json")
    manifest.record_file(source, selected, signature, output, "done")

    assert manifest.completed_output_for(source, selected, batch_signature({"runtime": {"provider": "cuda"}}, selected)) == ""
    source.write_bytes(b"changed")
    assert manifest.completed_output_for(source, selected, signature) == ""


def test_batch_resume_manifest_does_not_resume_corrupt_or_failed_reports(tmp_path: Path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    selected = [candidate("one", tmp_path)]
    signature = batch_signature({"runtime": {"provider": "cpu"}}, selected)
    manifest = BatchResumeManifest(tmp_path / "Logs" / "batch_resume_manifest.json")

    corrupt = tmp_path / "Output" / "corrupt"
    corrupt.mkdir(parents=True)
    (corrupt / "results.json").write_text("{not json", encoding="utf-8")
    manifest.record_file(source, selected, signature, corrupt, "done")
    assert manifest.completed_output_for(source, selected, signature) == ""

    failed = tmp_path / "Output" / "failed"
    failed.mkdir(parents=True)
    (failed / "results.json").write_text('{"runs":[],"errors":[{"status":"failed_before_model_run"}]}', encoding="utf-8")
    manifest.record_file(source, selected, signature, failed, "done")
    assert manifest.completed_output_for(source, selected, signature) == ""


def args_for(tmp_path: Path) -> Namespace:
    return Namespace(
        paths=[str(tmp_path / "a.wav"), str(tmp_path / "b.wav")],
        config=str(tmp_path / "config.json"),
        interactive=False,
        scan_only=False,
        doctor=False,
        json=False,
        strict=False,
        repair_plan=False,
        repair_all_safe=False,
        validate_real_smoke=False,
        install_deps=False,
        allow_downloads=False,
        no_network=False,
        full_real_smoke=False,
        first_run=False,
        first_run_smoke=False,
        download_model=False,
        download_model_first=False,
        open_models=False,
        open_input=False,
        watch=False,
        once=False,
    )


def test_batch_loop_writes_partial_summary_on_interrupt(monkeypatch, tmp_path: Path):
    from app import main

    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    model = candidate("model", tmp_path)
    output = tmp_path / "Output" / "a"
    output.mkdir(parents=True)
    (output / "results.json").write_text("{}", encoding="utf-8")
    config = {
        "folders": {"models": str(tmp_path / "Models"), "input": str(tmp_path / "Input"), "output": str(tmp_path / "Output"), "temp": str(tmp_path / "Temp"), "logs": str(tmp_path / "Logs"), "cache": str(tmp_path / "Cache")},
        "input": {"extensions": [".wav"], "recursive_folders": True, "file_stability_wait_seconds": 0},
        "runtime": {},
        "advanced": {"keep_temp_wavs": False},
    }
    summaries = []
    calls = {"count": 0}

    monkeypatch.setattr("app.main.load_config", lambda path: config)
    monkeypatch.setattr("app.main.scan_models", lambda root: ([model], []))
    monkeypatch.setattr("app.main.setup_logging", lambda path: None)
    monkeypatch.setattr("app.main.warn_runtime_dependency_fallbacks", lambda config: None)
    monkeypatch.setattr("app.main.ensure_dependencies", lambda selected, config, reference_llm=None: (selected, reference_llm))
    monkeypatch.setattr("app.main.collect_input_files", lambda args, config: [a, b])
    monkeypatch.setattr("app.main.write_batch_summary", lambda config, rows: summaries.append(list(rows)))

    def fake_process(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return output
        raise KeyboardInterrupt

    monkeypatch.setattr("app.main.process_file_with_candidates", fake_process)

    try:
        main._main(args_for(tmp_path))
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("expected KeyboardInterrupt")

    assert len(summaries) == 1
    assert summaries[0][0]["source_path"] == str(a.resolve())
