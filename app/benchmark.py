from __future__ import annotations

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
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def peak_vram_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
    except Exception:
        return None
    return None


@contextmanager
def timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
