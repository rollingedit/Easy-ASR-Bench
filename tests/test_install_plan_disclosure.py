from pathlib import Path

from app.install_plan import build_install_plan, format_install_plan


def test_install_plan_discloses_packages_indexes_and_fallback(tmp_path: Path):
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements" / "faster_whisper.txt").write_text("faster-whisper\nctranslate2\n", encoding="utf-8")
    config = {"runtime": {"provider": "cpu", "prefer_gpu": False}, "dependency_install": {}}

    plan = build_install_plan("faster_whisper", tmp_path, config, ["tiny-ct2"])
    text = format_install_plan(plan)

    assert plan.dependency_group == "faster_whisper"
    assert "faster-whisper" in plan.packages
    assert "ctranslate2" in plan.packages
    assert plan.confirmation_prompt == "Press Enter to install, or type s to skip this group."
    assert "Only models requiring this dependency group are skipped" in plan.fallback_if_declined
    assert "Network destinations" in text


def test_cuda_install_plan_discloses_large_accelerator_without_typed_confirmation(tmp_path: Path, monkeypatch):
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements" / "faster_whisper_cuda.txt").write_text("nvidia-cublas-cu12\n", encoding="utf-8")
    monkeypatch.setattr("app.dependency_manager.nvidia_gpu_detected", lambda: True)
    config = {
        "runtime": {"provider": "cuda", "prefer_gpu": True},
        "dependency_install": {"allow_cuda_install": True, "allow_accelerator_install": True},
    }

    plan = build_install_plan("faster_whisper", tmp_path, config, ["large-v3"])

    assert "INSTALL CUDA" not in format_install_plan(plan)
    assert "Press Enter to install" in format_install_plan(plan)
    assert plan.estimated_size_class == "large"
