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


def test_installer_creates_and_removes_start_menu_shortcuts():
    text = Path("installer/install.ps1").read_text(encoding="utf-8")

    assert "function Install-Shortcuts" in text
    assert "function Remove-Shortcuts" in text
    assert 'Microsoft\\Windows\\Start Menu\\Programs\\Easy ASR Bench' in text
    assert "New-Object -ComObject WScript.Shell" in text
    assert "Run Easy ASR Bench" in text
    assert "Drop Audio Or Folders" in text
    assert "Open Latest Report" in text
    assert "Open Output Folder" in text
    assert "Edit Config" in text
    assert "Repair Easy ASR Bench" in text
    assert "Uninstall Easy ASR Bench" in text
