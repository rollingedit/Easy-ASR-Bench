import sys
from types import SimpleNamespace

from app.ctranslate2_probe import ctranslate2_cuda_available


def test_ctranslate2_cuda_probe_uses_device_count(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 1))

    assert ctranslate2_cuda_available() is True


def test_ctranslate2_cuda_probe_fails_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: (_ for _ in ()).throw(RuntimeError("no cuda"))))

    assert ctranslate2_cuda_available() is False
