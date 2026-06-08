from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

try:
    import psutil
except ModuleNotFoundError:  # Allows scanner/bootstrap commands before full dependency install.
    psutil = None


@dataclass
class VariantMetrics:
    variant: str
    precision: str
    provider: str
    audio_seconds: float
    chunk_count: int
    model_load_seconds: float = 0.0
    preprocessing_seconds: float = 0.0
    inference_seconds: float = 0.0
    total_wall_seconds: float = 0.0
    tokens_generated: int = 0
    peak_process_memory_mb: float = 0.0
    errors: int = 0
    transcript_path: str = ""
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        return self.tokens_generated / self.inference_seconds if self.inference_seconds > 0 else 0.0

    @property
    def real_time_factor(self) -> float:
        return self.inference_seconds / self.audio_seconds if self.audio_seconds > 0 else 0.0

    @property
    def audio_seconds_per_wall_second(self) -> float:
        return self.audio_seconds / self.total_wall_seconds if self.total_wall_seconds > 0 else 0.0


def process_memory_mb() -> float:
    if psutil is None:
        return 0.0
    return psutil.Process().memory_info().rss / (1024 * 1024)


def reset_peak_vram() -> None:
    _WINDOWS_VRAM_MONITOR.start()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def peak_vram_mb() -> float | None:
    sample = peak_vram_sample()
    return sample["peak_vram_mb"]


def peak_vram_sample() -> dict[str, float | str | None]:
    windows_sample = _WINDOWS_VRAM_MONITOR.stop()
    torch_peak = _torch_peak_vram_mb()
    windows_peak = windows_sample.peak_dedicated_mb
    if windows_peak is not None:
        return {
            "peak_vram_mb": windows_peak,
            "vram_measurement_source": windows_sample.source,
            "torch_peak_vram_mb": torch_peak,
            "windows_peak_dedicated_vram_mb": windows_peak,
            "vram_measurement_note": "Windows GPU Adapter Memory dedicated usage is vendor-neutral and can include NVIDIA, AMD, and Intel adapters.",
        }
    if torch_peak is not None:
        return {
            "peak_vram_mb": torch_peak,
            "vram_measurement_source": "torch_cuda_allocator",
            "torch_peak_vram_mb": torch_peak,
            "windows_peak_dedicated_vram_mb": None,
            "vram_measurement_note": "Torch CUDA allocator peak only covers PyTorch CUDA allocations; non-Torch backends may allocate GPU memory outside this number.",
        }
    return {
        "peak_vram_mb": None,
        "vram_measurement_source": "unavailable",
        "torch_peak_vram_mb": None,
        "windows_peak_dedicated_vram_mb": None,
        "vram_measurement_note": "VRAM telemetry was unavailable from Windows GPU counters and Torch CUDA.",
    }


def _torch_peak_vram_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
    except Exception:
        return None
    return None


@dataclass
class WindowsVramSample:
    peak_dedicated_mb: float | None = None
    source: str = "unavailable"


class WindowsVramMonitor:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peak_dedicated_mb: float | None = None
        self._source = "unavailable"

    def start(self) -> None:
        self.stop()
        with self._lock:
            self._peak_dedicated_mb = _windows_dedicated_vram_usage_mb()
            self._source = "windows_gpu_adapter_memory" if self._peak_dedicated_mb is not None else "unavailable"
        if os.name != "nt":
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> WindowsVramSample:
        thread = self._thread
        if thread is not None:
            self._stop.set()
            thread.join(timeout=2.0)
            self._thread = None
        latest = _windows_dedicated_vram_usage_mb()
        with self._lock:
            if latest is not None:
                self._source = "windows_gpu_adapter_memory"
                self._peak_dedicated_mb = max(self._peak_dedicated_mb or 0.0, latest)
            return WindowsVramSample(self._peak_dedicated_mb, self._source)

    def _poll(self) -> None:
        while not self._stop.wait(0.5):
            sample = _windows_dedicated_vram_usage_mb()
            if sample is None:
                continue
            with self._lock:
                self._source = "windows_gpu_adapter_memory"
                self._peak_dedicated_mb = max(self._peak_dedicated_mb or 0.0, sample)


def _windows_dedicated_vram_usage_mb() -> float | None:
    if os.name != "nt":
        return None
    sample = _windows_gpu_adapter_memory_via_powershell()
    if sample is not None:
        return sample
    return _windows_gpu_adapter_memory_via_typeperf()


def _windows_gpu_adapter_memory_via_powershell() -> float | None:
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "$c = Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -ErrorAction Stop; "
        "($c.CounterSamples | Measure-Object -Property CookedValue -Sum).Sum",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=4)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return _bytes_text_to_mb(completed.stdout)


def _windows_gpu_adapter_memory_via_typeperf() -> float | None:
    command = ["typeperf", r"\GPU Adapter Memory(*)\Dedicated Usage", "-sc", "1"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=4)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    values = [float(match.group(1)) for match in re.finditer(r'"(-?\d+(?:\.\d+)?)"', completed.stdout)]
    if not values:
        return None
    return max(0.0, sum(values) / (1024 * 1024))


def _bytes_text_to_mb(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    value = float(match.group(0))
    if value < 0:
        return None
    return value / (1024 * 1024)


_WINDOWS_VRAM_MONITOR = WindowsVramMonitor()


@contextmanager
def timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
