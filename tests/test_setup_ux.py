from pathlib import Path


def test_setup_offers_run_now_after_public_install():
    setup = Path("setup.bat").read_text(encoding="utf-8")

    assert "Setup complete." in setup
    assert "[R] Run Easy ASR Bench now" in setup
    assert "[P] Paste a Hugging Face model link to download" in setup
    assert "[M] Open Models folder" in setup
    assert "[I] Open Input folder" in setup
    assert 'choice /C RPMIQ /N /M "Choose R, P, M, I, or Q: "' in setup
    assert "Run.bat" in setup
    assert "--first-run" in setup
    assert "--download-model" in setup


def test_setup_winget_accepts_required_agreements():
    setup = Path("setup.bat").read_text(encoding="utf-8")

    assert "winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements" in setup
