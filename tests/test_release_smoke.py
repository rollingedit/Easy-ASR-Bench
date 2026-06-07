import json
import zipfile
from pathlib import Path

import pytest

import scripts.write_release_smoke as write_release_smoke
from scripts.verify_github_release import download_assets, sha256, verify_release, write_transcript


def test_write_release_smoke_records_automated_passes_and_manual_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(write_release_smoke, "ROOT", tmp_path)
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "Easy-ASR-Bench-v0.3.1-win.zip").write_bytes(b"zip")

    monkeypatch.setattr(write_release_smoke, "validate_version_coherence", lambda tag: None)
    monkeypatch.setattr(write_release_smoke, "verify_assets", lambda tag: (True, {"setup.bat": "sha256:abc"}))
    monkeypatch.setattr(write_release_smoke, "current_commit", lambda: "abc123")
    monkeypatch.setattr(
        write_release_smoke,
        "run_check",
        lambda name, command: {"name": name, "status": "pass", "command": " ".join(command), "exit_code": 0},
    )

    output = tmp_path / "release-smoke-v0.3.1.json"
    write_release_smoke.write_smoke("v0.3.1", output)
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["schema"] == "easy_asr_bench.release_smoke.v2"
    assert {"id": "win11_clean_no_python_setup", "status": "not_run"} in data["manual_rows"]
    assert data["tag"] == "v0.3.1"
    assert data["commit"] == "abc123"
    assert data["asset_hashes_verified"] is True
    assert all(check["status"] == "pass" for check in data["checks"])
    assert data["manual_matrix"]["win11_clean_no_python_setup"] == "not_run"
    assert data["manual_matrix"]["win10_existing_python_setup"] == "not_run"
    assert data["manual_matrix"]["provider_smoke"]["nvidia_cuda_torch_onnx_faster_whisper_llama"] == "not_run"
    assert data["manual_matrix"]["model_smoke"]["gguf_reference_llm"] == "not_run"
    assert data["manual_matrix"]["model_smoke"]["audio_asr_gguf_mmproj"] == "not_run"


def test_write_release_smoke_fails_when_automated_check_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(write_release_smoke, "ROOT", tmp_path)
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "Easy-ASR-Bench-v0.3.1-win.zip").write_bytes(b"zip")

    monkeypatch.setattr(write_release_smoke, "validate_version_coherence", lambda tag: None)
    monkeypatch.setattr(write_release_smoke, "verify_assets", lambda tag: (True, {"setup.bat": "sha256:abc"}))
    monkeypatch.setattr(
        write_release_smoke,
        "run_check",
        lambda name, command: {"name": name, "status": "fail", "command": " ".join(command), "exit_code": 1},
    )

    with pytest.raises(SystemExit, match="release smoke checks failed"):
        write_release_smoke.write_smoke("v0.3.1", tmp_path / "release-smoke-v0.3.1.json")


def test_write_release_smoke_fails_when_asset_hashes_do_not_match(tmp_path, monkeypatch):
    monkeypatch.setattr(write_release_smoke, "ROOT", tmp_path)
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "Easy-ASR-Bench-v0.3.1-win.zip").write_bytes(b"zip")

    monkeypatch.setattr(write_release_smoke, "validate_version_coherence", lambda tag: None)
    monkeypatch.setattr(write_release_smoke, "verify_assets", lambda tag: (False, {"setup.bat": "sha256:bad"}))

    with pytest.raises(SystemExit, match="asset hashes do not match"):
        write_release_smoke.write_smoke("v0.3.1", tmp_path / "release-smoke-v0.3.1.json")


def test_verify_github_release_requires_smoke_asset_for_v2(tmp_path, monkeypatch):
    release = {
        "assets": [
            {"name": "setup.bat", "browser_download_url": "https://example/setup.bat"},
            {"name": "install.ps1", "browser_download_url": "https://example/install.ps1"},
            {"name": "manifest.json", "browser_download_url": "https://example/manifest.json"},
            {"name": "checksums.json", "browser_download_url": "https://example/checksums.json"},
            {"name": "Easy-ASR-Bench-v0.3.1-win.zip", "browser_download_url": "https://example/app.zip"},
        ]
    }

    def fake_request_json(url):
        if url.endswith("/releases/tags/v0.3.1"):
            return release
        return {"object": {"type": "commit", "sha": "abc123"}}

    def fake_download(url, destination: Path):
        if destination.name == "manifest.json":
            destination.write_text(
                json.dumps(
                    {
                        "schema": "easy_asr_bench.installer_manifest.v2",
                        "tag": "v0.3.1",
                        "app_zip": "Easy-ASR-Bench-v0.3.1-win.zip",
                        "installer_asset": "install.ps1",
                    }
                ),
                encoding="utf-8",
            )
        elif destination.name == "checksums.json":
            destination.write_text(json.dumps({"files": {}}), encoding="utf-8")
        else:
            destination.write_bytes(b"asset")

    monkeypatch.setattr("scripts.verify_github_release.tempfile.mkdtemp", lambda prefix: str(tmp_path))
    monkeypatch.setattr("scripts.verify_github_release.request_json", fake_request_json)
    monkeypatch.setattr("scripts.verify_github_release.download", fake_download)

    with pytest.raises(AssertionError, match="Release smoke asset is missing"):
        verify_release("owner/repo", "v0.3.1", "abc123")


def test_verify_github_release_accepts_complete_mocked_release_with_smoke(tmp_path, monkeypatch):
    asset_bytes: dict[str, bytes] = {}
    zip_path = tmp_path / "Easy-ASR-Bench-v0.3.1-win.zip"
    app_root = tmp_path / "zip-root" / "Easy-ASR-Bench-v0.3.1"
    app_root.mkdir(parents=True)
    (app_root / "setup.bat").write_text("@echo off\r\n" * 200, encoding="utf-8", newline="")
    (app_root / "Run.bat").write_text("@echo off\r\necho run\r\n" * 5, encoding="utf-8", newline="")
    (app_root / "Drop_Audio_Or_Folders_Here.bat").write_text("@echo off\r\necho drop\r\n" * 5, encoding="utf-8", newline="")
    (app_root / "installer").mkdir()
    (app_root / "installer" / "install.ps1").write_text("Write-Host ok\r\n" * 250, encoding="utf-8", newline="")
    (app_root / "scripts").mkdir()
    (app_root / "scripts" / "validate_physical_files.py").write_text("print('ok')\n" * 150, encoding="utf-8", newline="\n")
    (app_root / "scripts" / "verify_github_release.py").write_text("print('ok')\n" * 150, encoding="utf-8", newline="\n")
    (app_root / ".github" / "workflows").mkdir(parents=True)
    (app_root / ".github" / "workflows" / "release-gate.yml").write_text("name: x\n" * 75, encoding="utf-8", newline="\n")
    (app_root / ".github" / "workflows" / "publish-release.yml").write_text("name: x\n" * 50, encoding="utf-8", newline="\n")
    (app_root / "app").mkdir()
    (app_root / "app" / "model_scanner.py").write_text("print('ok')\n" * 600, encoding="utf-8", newline="\n")
    (app_root / "app" / "hf_model_downloader.py").write_text("print('ok')\n" * 300, encoding="utf-8", newline="\n")
    (app_root / "app" / "results_writer.py").write_text("print('ok')\n" * 100, encoding="utf-8", newline="\n")
    (app_root / "app" / "scoring.py").write_text("print('ok')\n" * 80, encoding="utf-8", newline="\n")
    (app_root / "app" / "main.py").write_text("print('ok')\n" * 100, encoding="utf-8", newline="\n")
    (app_root / "requirements").mkdir()
    (app_root / "requirements" / "core.txt").write_text("numpy\nsoundfile\nlibrosa\npsutil\njiwer\n", encoding="utf-8", newline="\n")
    (app_root / "config.json").write_text(
        json.dumps(
            {
                "app": {"version": "0.3.1"},
                "folders": {
                    "models": "Models",
                    "input": "Input",
                    "output": "Output",
                    "temp": "Temp",
                    "logs": "Logs",
                    "cache": "Cache",
                },
                "input": {
                    "extensions": [".wav"],
                    "recursive_folders": True,
                    "file_stability_wait_seconds": 0,
                },
                "runtime": {"provider": "auto"},
                "advanced": {"keep_temp_wavs": False},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in app_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(tmp_path / "zip-root").as_posix())

    asset_bytes["setup.bat"] = b"setup"
    asset_bytes["install.ps1"] = b"install"
    asset_bytes["manifest.json"] = json.dumps(
        {
            "schema": "easy_asr_bench.installer_manifest.v2",
            "tag": "v0.3.1",
            "version": "0.3.1",
            "app_zip": "Easy-ASR-Bench-v0.3.1-win.zip",
            "installer_asset": "install.ps1",
        }
    ).encode()
    asset_bytes["Easy-ASR-Bench-v0.3.1-win.zip"] = zip_path.read_bytes()
    checksums = {
        "files": {
            "setup.bat": "sha256:" + __import__("hashlib").sha256(asset_bytes["setup.bat"]).hexdigest(),
            "install.ps1": "sha256:" + __import__("hashlib").sha256(asset_bytes["install.ps1"]).hexdigest(),
            "manifest.json": "sha256:" + __import__("hashlib").sha256(asset_bytes["manifest.json"]).hexdigest(),
            "Easy-ASR-Bench-v0.3.1-win.zip": "sha256:" + __import__("hashlib").sha256(asset_bytes["Easy-ASR-Bench-v0.3.1-win.zip"]).hexdigest(),
        }
    }
    asset_bytes["checksums.json"] = json.dumps(checksums).encode()
    asset_bytes["release-smoke-v0.3.1.json"] = json.dumps(
        {
            "schema": "easy_asr_bench.release_smoke.v2",
            "tag": "v0.3.1",
            "commit": "abc123",
            "asset_hashes_verified": True,
            "manual_rows": [{"id": "win11_clean_no_python_setup", "status": "not_run"}],
            "manual_matrix": {"win11_clean_no_python_setup": "not_run"},
        }
    ).encode()
    release = {
        "assets": [
            {"name": name, "browser_download_url": f"https://example/{name}"}
            for name in asset_bytes
        ]
    }

    def fake_request_json(url):
        if url.endswith("/releases/tags/v0.3.1"):
            return release
        return {"object": {"type": "commit", "sha": "abc123"}}

    def fake_download(url, destination: Path):
        destination.write_bytes(asset_bytes[destination.name])

    monkeypatch.setattr("scripts.verify_github_release.request_json", fake_request_json)
    monkeypatch.setattr("scripts.verify_github_release.download", fake_download)

    verify_release("owner/repo", "v0.3.1", "abc123")


def test_write_release_verification_transcript_records_assets(tmp_path):
    transcript = tmp_path / "release-verification-v0.3.1.txt"

    write_transcript(
        transcript,
        repo="owner/repo",
        tag="v0.3.1",
        expected_commit="abc123",
        resolved_commit="abc123",
        release={"id": 12, "draft": True, "prerelease": False},
        hashes={"setup.bat": "sha256:aaa", "Easy-ASR-Bench-v0.3.1-win.zip": "sha256:bbb"},
        zip_name="Easy-ASR-Bench-v0.3.1-win.zip",
    )

    text = transcript.read_text(encoding="utf-8")
    assert "Easy ASR Bench release verification transcript" in text
    assert "resolved_release_commit: abc123" in text
    assert "setup.bat sha256:aaa" in text
    assert "zip_physical_validation: pass" in text
    assert "not marked pass" in text


def test_download_assets_prefers_authenticated_asset_api_url(tmp_path, monkeypatch):
    requested: list[str] = []

    def fake_download(url, destination: Path):
        requested.append(url)
        destination.write_bytes(b"asset")

    monkeypatch.setattr("scripts.verify_github_release.download", fake_download)

    local = download_assets(
        {
            "setup.bat": {
                "url": "https://api.github.com/repos/owner/repo/releases/assets/1",
                "browser_download_url": "https://github.com/owner/repo/releases/download/v1/setup.bat",
            }
        },
        tmp_path,
    )

    assert requested == ["https://api.github.com/repos/owner/repo/releases/assets/1"]
    assert local["setup.bat"].read_bytes() == b"asset"


def test_verify_github_release_rejects_tampered_asset_hash(tmp_path, monkeypatch):
    release = {
        "assets": [
            {"name": "setup.bat", "browser_download_url": "https://example/setup.bat"},
            {"name": "install.ps1", "browser_download_url": "https://example/install.ps1"},
            {"name": "manifest.json", "browser_download_url": "https://example/manifest.json"},
            {"name": "checksums.json", "browser_download_url": "https://example/checksums.json"},
            {"name": "Easy-ASR-Bench-v0.3.1-win.zip", "browser_download_url": "https://example/app.zip"},
            {"name": "release-smoke-v0.3.1.json", "browser_download_url": "https://example/smoke.json"},
        ]
    }

    def fake_request_json(url):
        if url.endswith("/releases/tags/v0.3.1"):
            return release
        return {"object": {"type": "commit", "sha": "abc123"}}

    def fake_download(url, destination: Path):
        if destination.name == "manifest.json":
            destination.write_text(
                json.dumps(
                    {
                        "schema": "easy_asr_bench.installer_manifest.v2",
                        "tag": "v0.3.1",
                        "app_zip": "Easy-ASR-Bench-v0.3.1-win.zip",
                        "installer_asset": "install.ps1",
                    }
                ),
                encoding="utf-8",
            )
        elif destination.name == "checksums.json":
            destination.write_text(json.dumps({"files": {"setup.bat": "sha256:not-real"}}), encoding="utf-8")
        elif destination.name == "release-smoke-v0.3.1.json":
            destination.write_text(
                json.dumps(
                    {
                        "schema": "easy_asr_bench.release_smoke.v2",
                        "tag": "v0.3.1",
                        "commit": "abc123",
                        "asset_hashes_verified": True,
                        "manual_rows": [{"id": "win11_clean_no_python_setup", "status": "not_run"}],
                        "manual_matrix": {"win11_clean_no_python_setup": "not_run"},
                    }
                ),
                encoding="utf-8",
            )
        else:
            destination.write_bytes(b"asset")

    monkeypatch.setattr("scripts.verify_github_release.tempfile.mkdtemp", lambda prefix: str(tmp_path))
    monkeypatch.setattr("scripts.verify_github_release.request_json", fake_request_json)
    monkeypatch.setattr("scripts.verify_github_release.download", fake_download)

    with pytest.raises(AssertionError, match="Checksum mismatch for setup.bat"):
        verify_release("owner/repo", "v0.3.1", "abc123")


def test_verify_github_release_requires_explicit_manual_rows_in_v2_smoke(tmp_path, monkeypatch):
    release = {
        "assets": [
            {"name": "setup.bat", "browser_download_url": "https://example/setup.bat"},
            {"name": "install.ps1", "browser_download_url": "https://example/install.ps1"},
            {"name": "manifest.json", "browser_download_url": "https://example/manifest.json"},
            {"name": "checksums.json", "browser_download_url": "https://example/checksums.json"},
            {"name": "Easy-ASR-Bench-v0.3.1-win.zip", "browser_download_url": "https://example/app.zip"},
            {"name": "release-smoke-v0.3.1.json", "browser_download_url": "https://example/smoke.json"},
        ]
    }

    def fake_request_json(url):
        if url.endswith("/releases/tags/v0.3.1"):
            return release
        return {"object": {"type": "commit", "sha": "abc123"}}

    def fake_download(url, destination: Path):
        if destination.name == "manifest.json":
            destination.write_text(
                json.dumps(
                    {
                        "schema": "easy_asr_bench.installer_manifest.v2",
                        "tag": "v0.3.1",
                        "app_zip": "Easy-ASR-Bench-v0.3.1-win.zip",
                        "installer_asset": "install.ps1",
                    }
                ),
                encoding="utf-8",
            )
        elif destination.name == "release-smoke-v0.3.1.json":
            destination.write_text(
                json.dumps(
                    {
                        "schema": "easy_asr_bench.release_smoke.v2",
                        "tag": "v0.3.1",
                        "commit": "abc123",
                        "asset_hashes_verified": True,
                    }
                ),
                encoding="utf-8",
            )
        elif destination.name == "checksums.json":
            destination.write_text(json.dumps({"files": {}}), encoding="utf-8")
        else:
            destination.write_bytes(b"asset")

    monkeypatch.setattr("scripts.verify_github_release.tempfile.mkdtemp", lambda prefix: str(tmp_path))
    monkeypatch.setattr("scripts.verify_github_release.request_json", fake_request_json)
    monkeypatch.setattr("scripts.verify_github_release.download", fake_download)

    with pytest.raises(AssertionError, match="must include explicit manual_rows"):
        verify_release("owner/repo", "v0.3.1", "abc123")
