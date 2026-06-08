import subprocess

from app.ctranslate2_probe import ctranslate2_cuda_available, ctranslate2_probe


def completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["python"], returncode, stdout=stdout, stderr=stderr)


def test_ctranslate2_cuda_probe_uses_isolated_process(monkeypatch):
    def fake_run(command, capture_output=True, text=True, timeout=15):
        assert command[:2]
        assert command[1] == "-c"
        return completed('{"ok": true, "cuda_available": true, "version": "4.4.0"}\n')

    monkeypatch.setattr("app.ctranslate2_probe.subprocess.run", fake_run)

    assert ctranslate2_cuda_available() is True


def test_ctranslate2_cuda_probe_fails_closed_on_native_crash(monkeypatch):
    def fake_run(command, capture_output=True, text=True, timeout=15):
        return completed(stderr="Windows fatal exception: access violation", returncode=3221225477)

    monkeypatch.setattr("app.ctranslate2_probe.subprocess.run", fake_run)

    result = ctranslate2_probe()

    assert ctranslate2_cuda_available() is False
    assert result["ok"] is False
    assert result["error_type"] == "NativeProbeFailed"
