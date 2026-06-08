import subprocess

import app.benchmark as benchmark


def completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["tool"], returncode, stdout=stdout, stderr=stderr)


def test_windows_counter_bytes_are_reported_as_mb(monkeypatch):
    monkeypatch.setattr(benchmark.os, "name", "nt")
    monkeypatch.setattr(benchmark, "_windows_gpu_adapter_memory_via_powershell", lambda: 1536.0)
    monkeypatch.setattr(benchmark, "_windows_gpu_adapter_memory_via_typeperf", lambda: None)

    assert benchmark._windows_dedicated_vram_usage_mb() == 1536.0


def test_typeperf_adapter_memory_sums_vendor_neutral_gpu_counters(monkeypatch):
    monkeypatch.setattr(benchmark.os, "name", "nt")

    def fake_run(command, capture_output=True, text=True, timeout=4):
        assert command[0] == "typeperf"
        return completed(
            stdout=(
                '"(PDH-CSV 4.0)","\\GPU Adapter Memory(luid_0x00000000_phys_0)\\Dedicated Usage",'
                '"\\GPU Adapter Memory(luid_0x00000001_phys_0)\\Dedicated Usage"\n'
                '"06/07/2026 12:00:00.000","1048576.000000","2097152.000000"\n'
            )
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)

    assert benchmark._windows_gpu_adapter_memory_via_typeperf() == 3.0


def test_peak_vram_sample_prefers_windows_gpu_adapter_memory(monkeypatch):
    monkeypatch.setattr(benchmark._WINDOWS_VRAM_MONITOR, "stop", lambda: benchmark.WindowsVramSample(2048.0, "windows_gpu_adapter_memory"))
    monkeypatch.setattr(benchmark, "_torch_peak_vram_mb", lambda: 512.0)

    sample = benchmark.peak_vram_sample()

    assert sample["peak_vram_mb"] == 2048.0
    assert sample["vram_measurement_source"] == "windows_gpu_adapter_memory"
    assert sample["torch_peak_vram_mb"] == 512.0
    assert "NVIDIA, AMD, and Intel" in sample["vram_measurement_note"]


def test_peak_vram_sample_labels_torch_only_fallback(monkeypatch):
    monkeypatch.setattr(benchmark._WINDOWS_VRAM_MONITOR, "stop", lambda: benchmark.WindowsVramSample(None, "unavailable"))
    monkeypatch.setattr(benchmark, "_torch_peak_vram_mb", lambda: 256.0)

    sample = benchmark.peak_vram_sample()

    assert sample["peak_vram_mb"] == 256.0
    assert sample["vram_measurement_source"] == "torch_cuda_allocator"
    assert "non-Torch backends" in sample["vram_measurement_note"]
