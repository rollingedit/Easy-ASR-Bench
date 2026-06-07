from pathlib import Path


def test_standalone_setup_can_use_installed_uninstaller():
    text = Path("setup.bat").read_text(encoding="utf-8")

    assert r"%INSTALL_DIR%\installer\install.ps1" in text
    assert "Installer script was not found next to setup.bat or in the installed app folder." in text


def test_destructive_uninstall_requires_explicit_confirmation():
    text = Path("installer/install.ps1").read_text(encoding="utf-8")

    assert "ConfirmRemoveUserData" in text
    assert "DELETE EASY ASR BENCH USER DATA" in text
    assert 'ConfirmRemoveUserData -ne "DELETE USER DATA"' not in text
    assert "Destructive uninstall refused" in text
