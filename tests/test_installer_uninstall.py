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


def test_installer_stashes_preserved_user_data_until_validation_succeeds():
    text = Path("installer/install.ps1").read_text(encoding="utf-8")

    assert '$Preserve = "$InstallDir.preserve"' in text
    assert "Move-PreservedUserData $Backup $Preserve" in text
    assert "Move-PreservedUserData $Backup $New" not in text
    assert "Move-PreservedUserData $Preserve $InstallDir" in text
    assert "Restoring preserved user data into validated install." in text


def test_installer_rolls_preserved_data_back_for_repair_failures():
    text = Path("installer/install.ps1").read_text(encoding="utf-8")

    local_setup_catch = text.split('Write-SetupLog "Local setup failed."', 1)[1].split("finally {", 1)[0]
    assert "if (Test-Path $Backup)" in local_setup_catch
    assert "Restore-MovedUserData $Preserve $Backup" in local_setup_catch
    assert "-not $Repair" not in local_setup_catch


def test_installer_preserved_directory_move_merges_existing_destinations():
    text = Path("installer/install.ps1").read_text(encoding="utf-8")

    helper = text.split("function Move-PreservedDirectory", 1)[1].split("function Move-PreservedUserData", 1)[0]
    assert "Get-ChildItem -LiteralPath $From -Force" in helper
    assert "Move-Item -LiteralPath $child.FullName -Destination $destChild" in helper
    assert "Remove-Item -LiteralPath $From -Recurse -Force" in helper
