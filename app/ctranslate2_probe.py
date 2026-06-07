from __future__ import annotations


def ctranslate2_cuda_available() -> bool:
    try:
        import ctranslate2
    except Exception:
        return False
    try:
        if hasattr(ctranslate2, "get_cuda_device_count"):
            return int(ctranslate2.get_cuda_device_count()) > 0
        if hasattr(ctranslate2, "get_supported_compute_types"):
            return bool(ctranslate2.get_supported_compute_types("cuda"))
    except Exception:
        return False
    return False
