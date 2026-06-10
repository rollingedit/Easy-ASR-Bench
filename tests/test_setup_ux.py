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
    assert "--download-model-first" in setup
    assert "Run.bat\" --download-model\"" not in setup


def test_setup_winget_accepts_required_agreements():
    setup = Path("setup.bat").read_text(encoding="utf-8")

    assert "winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements" in setup
    assert "where winget >nul 2>nul" in setup
    assert "Downloading Python 3.12.10 from python.org" in setup
    assert 'curl.exe --fail --location --connect-timeout 30 --max-time 300 --silent --show-error --output "!PYTHON_INSTALLER!" "!PYTHON_INSTALLER_URL!"' in setup
    assert "PYTHON_DOWNLOAD_OK" in setup
    assert "curl download did not produce a complete Python installer. Trying PowerShell..." in setup
    assert "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest" in setup


def test_setup_installs_vc_redist_without_winget_for_clean_windows():
    setup = Path("setup.bat").read_text(encoding="utf-8")

    assert "call :ensure_vc_redist" in setup
    assert "winget install -e --id Microsoft.VCRedist.2015+.x64 --accept-package-agreements --accept-source-agreements" in setup
    assert "https://aka.ms/vc14/vc_redist.x64.exe" in setup
    assert 'curl.exe --fail --location --connect-timeout 30 --max-time 300 --silent --show-error --output "!VC_REDIST_INSTALLER!" "!VC_REDIST_URL!"' in setup
    assert "VC_DOWNLOAD_OK" in setup
    assert '"!VC_REDIST_INSTALLER!" /install /quiet /norestart' in setup


def test_setup_doctor_forwards_json_and_strict_flags():
    setup = Path("setup.bat").read_text(encoding="utf-8")

    assert 'if /I "%~1"=="--doctor"' in setup
    assert 'if /I "%%~A"=="--json" (' in setup
    assert "set DOCTOR_ARGS=!DOCTOR_ARGS! --json" in setup
    assert "set SETUP_JSON=1" in setup
    assert 'if /I "%%~A"=="--strict" set DOCTOR_ARGS=!DOCTOR_ARGS! --strict' in setup
    assert 'if /I "%%~A"=="--repair-plan" set DOCTOR_ARGS=!DOCTOR_ARGS! --repair-plan' in setup
    assert 'if /I "%%~A"=="--repair-all-safe" set DOCTOR_ARGS=!DOCTOR_ARGS! --repair-all-safe' in setup
    assert 'if /I "%%~A"=="--validate-real-smoke" set DOCTOR_ARGS=!DOCTOR_ARGS! --validate-real-smoke' in setup
    assert 'if /I "%%~A"=="--repair-model-layouts" set DOCTOR_ARGS=!DOCTOR_ARGS! --repair-model-layouts' in setup
    assert 'if /I "%%~A"=="--install-deps" set DOCTOR_ARGS=!DOCTOR_ARGS! --install-deps' in setup
    assert 'if /I "%%~A"=="--allow-downloads" set DOCTOR_ARGS=!DOCTOR_ARGS! --allow-downloads' in setup
    assert 'if /I "%%~A"=="--no-network" set DOCTOR_ARGS=!DOCTOR_ARGS! --no-network' in setup
    assert 'if /I "%%~A"=="--full-real-smoke" set DOCTOR_ARGS=!DOCTOR_ARGS! --full-real-smoke' in setup
    assert "app.doctor --config config.json %DOCTOR_ARGS%" in setup
    assert 'app.doctor --config "%INSTALL_DIR%\\config.json" %DOCTOR_ARGS%' in setup
    assert "python -m app.doctor --config config.json %DOCTOR_ARGS%" in setup


def test_setup_runs_configured_update_check_after_config_init():
    setup = Path("setup.bat").read_text(encoding="utf-8")

    assert "Checking for updates..." in setup
    assert '".venv\\Scripts\\python.exe" -m app.update_check --config config.json --context setup' in setup


def test_setup_repair_forwards_repair_mode_to_public_installer():
    setup = Path("setup.bat").read_text(encoding="utf-8")

    assert "set INSTALLER_MODE_ARGS=" in setup
    assert 'if /I "%%~A"=="--repair" set INSTALLER_MODE_ARGS=!INSTALLER_MODE_ARGS! -Repair' in setup
    assert "%INSTALLER_MODE_ARGS%" in setup


def test_setup_lonely_installer_download_uses_temp_path_outside_same_block_expansion():
    setup = Path("setup.bat").read_text(encoding="utf-8")

    assert "set TEMP_INSTALLER_PS1=%TEMP%\\Easy-ASR-Bench-install-%APP_VERSION%.ps1" in setup
    assert "Invoke-WebRequest -Uri '%INSTALLER_URL%' -OutFile '%TEMP_INSTALLER_PS1%'" in setup
    assert 'set "INSTALLER_PS1=%TEMP_INSTALLER_PS1%"' in setup
    assert "set INSTALLER_PS1=%TEMP%\\Easy-ASR-Bench-install-%APP_VERSION%.ps1" not in setup
    assert "Invoke-WebRequest -Uri '%INSTALLER_URL%' -OutFile '%INSTALLER_PS1%'" not in setup


def test_setup_dry_run_json_emits_machine_readable_contract():
    setup = Path("setup.bat").read_text(encoding="utf-8")

    assert ":emit_dry_run_json" in setup
    assert "easy_asr_bench.setup_dry_run.v1" in setup
    assert "no_files_modified" in setup
    assert 'if "%SETUP_JSON%"=="1" call :emit_dry_run_json' in setup


def test_public_installer_runs_local_setup_without_post_setup_menu():
    setup = Path("setup.bat").read_text(encoding="utf-8")
    installer = Path("installer/install.ps1").read_text(encoding="utf-8")

    assert "set NO_POST_SETUP_MENU=0" in setup
    assert 'if /I "%%~A"=="--no-post-setup-menu" set NO_POST_SETUP_MENU=1' in setup
    assert 'if "%NO_POST_SETUP_MENU%"=="1" exit /b 0' in setup
    assert "setup.bat --local --no-post-setup-menu" in installer


def test_run_bat_forwards_release_qa_flags_without_interactive_banner():
    run_bat = Path("Run.bat").read_text(encoding="utf-8")
    main = Path("app/main.py").read_text(encoding="utf-8")

    assert 'if /I "%~1"=="--doctor"' in run_bat
    assert "shift /1" in run_bat
    assert "app.main --doctor %*" in run_bat
    assert 'parser.add_argument("--strict", action="store_true")' in main
    assert "strict=bool(args.strict)" in main
    assert 'if /I "%~1"=="--first-run-smoke" goto direct_app' in run_bat
    assert "app.main %*" in run_bat


def test_normal_launchers_do_not_run_doctor_wall():
    run_bat = Path("Run.bat").read_text(encoding="utf-8")
    drop_bat = Path("Drop_Audio_Or_Folders_Here.bat").read_text(encoding="utf-8")

    assert 'call "%~dp0setup.bat" --local --no-post-setup-menu' in run_bat
    assert 'call "%~dp0setup.bat" --local --no-post-setup-menu' in drop_bat
    assert '".venv\\Scripts\\python.exe" -m app.main --interactive %*' in run_bat
    assert '".venv\\Scripts\\python.exe" -m app.main --interactive %*' in drop_bat


def test_open_latest_report_prefers_final_results_then_compare_html():
    launcher = Path("Open_Latest_Report.bat").read_text(encoding="utf-8")

    assert "final_results.html" in launcher
    assert "compare.html" in launcher
    assert launcher.index("final_results.html") < launcher.index("compare.html")
    assert "No final_results.html or compare.html report was found under Output." in launcher


def test_single_file_runs_write_final_results_wrapper():
    writer = Path("app/results_writer.py").read_text(encoding="utf-8")

    assert "final_results.html" in writer
    assert "render_single_file_final_results" in writer
    assert "compare.html" in writer
