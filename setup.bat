@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set APP_NAME=Easy ASR Bench
set APP_VERSION=v0.4.0
set INSTALL_DIR=%LOCALAPPDATA%\Easy-ASR-Bench
set INSTALLER_PS1=%~dp0installer\install.ps1
set TEMP_INSTALLER_PS1=%TEMP%\Easy-ASR-Bench-install-%APP_VERSION%.ps1
set INSTALLER_URL=https://github.com/rollingedit/Easy-ASR-Bench/releases/download/%APP_VERSION%/install.ps1
set INSTALLER_SHA256=sha256:e26a8e76911058fc1fa045fd13a3e00fe3118a3eef60dbd491073d2d350bf911
set VERIFY_RELEASE=0
set ASSET_DIR=
set NEXT_IS_ASSET_DIR=0
set DOCTOR_ARGS=
set NO_POST_SETUP_MENU=0
set INSTALLER_MODE_ARGS=
set SETUP_JSON=0

for %%A in (%*) do (
  if "!NEXT_IS_ASSET_DIR!"=="1" (
    set "ASSET_DIR=%%~A"
    set NEXT_IS_ASSET_DIR=0
  ) else (
    if /I "%%~A"=="--verify-release" set VERIFY_RELEASE=1
    if /I "%%~A"=="--asset-dir" set NEXT_IS_ASSET_DIR=1
    if /I "%%~A"=="--json" (
      set DOCTOR_ARGS=!DOCTOR_ARGS! --json
      set SETUP_JSON=1
    )
    if /I "%%~A"=="--strict" set DOCTOR_ARGS=!DOCTOR_ARGS! --strict
    if /I "%%~A"=="--repair-plan" set DOCTOR_ARGS=!DOCTOR_ARGS! --repair-plan
    if /I "%%~A"=="--repair-all-safe" set DOCTOR_ARGS=!DOCTOR_ARGS! --repair-all-safe
    if /I "%%~A"=="--validate-real-smoke" set DOCTOR_ARGS=!DOCTOR_ARGS! --validate-real-smoke
    if /I "%%~A"=="--repair-model-layouts" set DOCTOR_ARGS=!DOCTOR_ARGS! --repair-model-layouts
    if /I "%%~A"=="--install-deps" set DOCTOR_ARGS=!DOCTOR_ARGS! --install-deps
    if /I "%%~A"=="--allow-downloads" set DOCTOR_ARGS=!DOCTOR_ARGS! --allow-downloads
    if /I "%%~A"=="--no-network" set DOCTOR_ARGS=!DOCTOR_ARGS! --no-network
    if /I "%%~A"=="--full-real-smoke" set DOCTOR_ARGS=!DOCTOR_ARGS! --full-real-smoke
    if /I "%%~A"=="--no-post-setup-menu" set NO_POST_SETUP_MENU=1
    if /I "%%~A"=="--repair" set INSTALLER_MODE_ARGS=!INSTALLER_MODE_ARGS! -Repair
  )
)

if /I "%~1"=="--dry-run" goto dry_run
if /I "%~2"=="--dry-run" goto dry_run
if /I "%~1"=="--doctor" (
  goto doctor
)
if /I "%~1"=="--uninstall" goto uninstall
if /I "%~1"=="--repair" goto bootstrap
if /I "%~1"=="--update" goto bootstrap
if /I "%~1"=="--local" goto local_setup
if exist "%~dp0app\main.py" goto local_setup

:bootstrap
echo %APP_NAME% installer
echo.
echo Install folder:
echo   %INSTALL_DIR%
echo.
echo Setup will download the verified app ZIP for %APP_VERSION%, install or repair
echo the app, create a Python virtual environment, and install core packages.
echo.

if not exist "%INSTALLER_PS1%" (
  echo Downloading installer script for %APP_VERSION%...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri '%INSTALLER_URL%' -OutFile '%TEMP_INSTALLER_PS1%'"
  if errorlevel 1 (
    echo Could not download installer script.
    pause
    exit /b 1
  )
  set "INSTALLER_PS1=%TEMP_INSTALLER_PS1%"
)

call :verify_sha "%INSTALLER_PS1%" "%INSTALLER_SHA256%" "installer/install.ps1"
if errorlevel 1 (
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER_PS1%" ^
  -InstallDir "%INSTALL_DIR%" ^
  -Version "%APP_VERSION%" ^
  %INSTALLER_MODE_ARGS%

if errorlevel 1 (
  echo Setup failed. Check %INSTALL_DIR%\Logs\setup.log
  pause
  exit /b 1
)

echo.
echo Setup complete.
echo.
echo Next step:
echo   [R] Run Easy ASR Bench now
echo   [P] Paste a Hugging Face model link to download
echo   [M] Open Models folder
echo   [I] Open Input folder
echo   [Q] Quit
choice /C RPMIQ /N /M "Choose R, P, M, I, or Q: "
if errorlevel 5 exit /b 0
if errorlevel 4 (
  if not exist "%INSTALL_DIR%\Input" mkdir "%INSTALL_DIR%\Input"
  explorer "%INSTALL_DIR%\Input"
  exit /b 0
)
if errorlevel 3 (
  if not exist "%INSTALL_DIR%\Models" mkdir "%INSTALL_DIR%\Models"
  explorer "%INSTALL_DIR%\Models"
  exit /b 0
)
if errorlevel 2 (
  start "Easy ASR Bench" cmd /k ""%INSTALL_DIR%\Run.bat" --first-run --download-model-first"
  exit /b 0
)
if exist "%INSTALL_DIR%\Run.bat" (
  start "Easy ASR Bench" cmd /k ""%INSTALL_DIR%\Run.bat" --first-run"
)
exit /b 0

:dry_run
echo %APP_NAME% setup dry run
echo Version: %APP_VERSION%
echo Install folder: %INSTALL_DIR%
echo Mode: validates installer inputs without changing files.
if not "%ASSET_DIR%"=="" (
  echo Staged release asset folder: %ASSET_DIR%
  if exist "%ASSET_DIR%\install.ps1" set "INSTALLER_PS1=%ASSET_DIR%\install.ps1"
)
if not exist "%INSTALLER_PS1%" (
  if "%VERIFY_RELEASE%"=="0" (
    echo Installer script was not found beside setup.bat.
    echo Standalone setup will download and verify it from:
    echo   %INSTALLER_URL%
    echo Use --dry-run --verify-release to validate public release assets.
    if "%SETUP_JSON%"=="1" call :emit_dry_run_json 0 standalone_download_available
    exit /b 0
  )
  echo Downloading installer script for release verification...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri '%INSTALLER_URL%' -OutFile '%TEMP_INSTALLER_PS1%'"
  if errorlevel 1 (
    echo Could not download installer script.
    if "%SETUP_JSON%"=="1" call :emit_dry_run_json 1 installer_download_failed
    exit /b 1
  )
  set "INSTALLER_PS1=%TEMP_INSTALLER_PS1%"
)
call :verify_sha "%INSTALLER_PS1%" "%INSTALLER_SHA256%" "installer/install.ps1"
if errorlevel 1 (
  if "%SETUP_JSON%"=="1" call :emit_dry_run_json 1 installer_sha_failed
  exit /b 1
)
if "%VERIFY_RELEASE%"=="1" (
  if "%ASSET_DIR%"=="" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER_PS1%" ^
      -InstallDir "%INSTALL_DIR%" ^
      -Version "%APP_VERSION%" ^
      -DryRun ^
      -VerifyRelease ^
      %INSTALLER_MODE_ARGS%
  ) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER_PS1%" ^
      -InstallDir "%INSTALL_DIR%" ^
      -Version "%APP_VERSION%" ^
      -DryRun ^
      -VerifyRelease ^
      -AssetDir "%ASSET_DIR%" ^
      %INSTALLER_MODE_ARGS%
  )
  set DRY_RUN_EXIT=!ERRORLEVEL!
  if "%SETUP_JSON%"=="1" call :emit_dry_run_json !DRY_RUN_EXIT! verify_release
  exit /b !DRY_RUN_EXIT!
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER_PS1%" ^
  -InstallDir "%INSTALL_DIR%" ^
  -Version "%APP_VERSION%" ^
  -DryRun ^
  %INSTALLER_MODE_ARGS%
set DRY_RUN_EXIT=!ERRORLEVEL!
if "%SETUP_JSON%"=="1" call :emit_dry_run_json !DRY_RUN_EXIT! local_or_standalone
exit /b !DRY_RUN_EXIT!

:doctor
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m app.doctor --config config.json %DOCTOR_ARGS%
  exit /b !ERRORLEVEL!
)
if exist "%INSTALL_DIR%\.venv\Scripts\python.exe" (
  "%INSTALL_DIR%\.venv\Scripts\python.exe" -m app.doctor --config "%INSTALL_DIR%\config.json" %DOCTOR_ARGS%
  exit /b !ERRORLEVEL!
)
if exist "%~dp0app\doctor.py" (
  python -m app.doctor --config config.json %DOCTOR_ARGS%
  exit /b !ERRORLEVEL!
)
echo Easy ASR Bench is not installed here and no installed runtime was found.
exit /b 1

:uninstall
if not exist "%INSTALLER_PS1%" (
  if exist "%INSTALL_DIR%\installer\install.ps1" (
    set INSTALLER_PS1=%INSTALL_DIR%\installer\install.ps1
  ) else (
    echo Installer script was not found next to setup.bat or in the installed app folder.
    echo Run setup.bat first, or uninstall from the installed folder.
    pause
    exit /b 1
  )
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER_PS1%" ^
  -InstallDir "%INSTALL_DIR%" ^
  -Uninstall
exit /b !ERRORLEVEL!

:local_setup
echo %APP_NAME% - local setup
echo.

set PYEXE=
for %%V in (3.14 3.13 3.12 3.11 3.10) do (
  if "!PYEXE!"=="" (
    py -%%V -c "import sys" >nul 2>nul
    if not errorlevel 1 set PYEXE=py -%%V
  )
)

if "%PYEXE%"=="" (
  echo Python 3.10 through 3.14 was not found. Attempting install with winget...
  where winget >nul 2>nul
  if not errorlevel 1 (
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
  ) else (
    echo winget was not found. Downloading Python 3.12.10 from python.org...
    set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
    set "PYTHON_INSTALLER=%TEMP%\python-3.12.10-amd64.exe"
    set PYTHON_DOWNLOAD_OK=0
    curl.exe --fail --location --connect-timeout 30 --max-time 300 --silent --show-error --output "!PYTHON_INSTALLER!" "!PYTHON_INSTALLER_URL!"
    if exist "!PYTHON_INSTALLER!" (
      for %%F in ("!PYTHON_INSTALLER!") do if %%~zF GTR 1000000 set PYTHON_DOWNLOAD_OK=1
    )
    if "!PYTHON_DOWNLOAD_OK!"=="0" (
      echo curl download did not produce a complete Python installer. Trying PowerShell...
      powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '!PYTHON_INSTALLER_URL!' -OutFile '!PYTHON_INSTALLER!'"
    )
    set PYTHON_DOWNLOAD_OK=0
    if exist "!PYTHON_INSTALLER!" (
      for %%F in ("!PYTHON_INSTALLER!") do if %%~zF GTR 1000000 set PYTHON_DOWNLOAD_OK=1
    )
    if "!PYTHON_DOWNLOAD_OK!"=="0" (
      echo Python installer download failed. Install Python 3.12 and rerun setup.bat.
      pause
      exit /b 1
    )
    "!PYTHON_INSTALLER!" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1 SimpleInstall=1
    if errorlevel 1 (
      echo Python installer failed. Install Python 3.12 and rerun setup.bat.
      pause
      exit /b 1
    )
  )
  py -3.12 -c "import sys" >nul 2>nul
  if not errorlevel 1 set PYEXE=py -3.12
  if "!PYEXE!"=="" (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
    if not errorlevel 1 set PYEXE=python
  )
  if "!PYEXE!"=="" (
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PYEXE="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  )
  if "!PYEXE!"=="" (
    echo Python install was not detected. Install Python 3.12 and rerun setup.bat.
    pause
    exit /b 1
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PYEXE% -m venv .venv
  if errorlevel 1 (
    pause
    exit /b 1
  )
)

echo Installing core Python packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements\core.txt
if errorlevel 1 (
  pause
  exit /b 1
)

echo Creating config if needed...
".venv\Scripts\python.exe" -m app.config --init
if errorlevel 1 (
  pause
  exit /b 1
)

echo Running doctor...
".venv\Scripts\python.exe" -m app.doctor --config config.json

echo Running self-test...
".venv\Scripts\python.exe" -m app.self_test --config config.json
if errorlevel 1 (
  pause
  exit /b 1
)

if "%NO_POST_SETUP_MENU%"=="1" exit /b 0

echo.
echo Setup complete.
echo.
echo Next step:
echo   [R] Run Easy ASR Bench now
echo   [P] Paste a Hugging Face model link to download
echo   [M] Open Models folder
echo   [I] Open Input folder
echo   [Q] Quit
choice /C RPMIQ /N /M "Choose R, P, M, I, or Q: "
if errorlevel 5 exit /b 0
if errorlevel 4 (
  if not exist "%CD%\Input" mkdir "%CD%\Input"
  explorer "%CD%\Input"
  exit /b 0
)
if errorlevel 3 (
  if not exist "%CD%\Models" mkdir "%CD%\Models"
  explorer "%CD%\Models"
  exit /b 0
)
if errorlevel 2 (
  call "%~dp0Run.bat" --first-run --download-model-first
  exit /b 0
)
call "%~dp0Run.bat" --first-run
exit /b 0

:verify_sha
set VERIFY_FILE=%~1
set EXPECTED_SHA=%~2
set VERIFY_LABEL=%~3
set ACTUAL_SHA=
for /f "tokens=1" %%H in ('certutil -hashfile "%VERIFY_FILE%" SHA256 ^| findstr /R "^[0-9A-Fa-f][0-9A-Fa-f]*$"') do (
  set ACTUAL_SHA=sha256:%%H
)
if "%ACTUAL_SHA%"=="" (
  echo Could not compute SHA256 for %VERIFY_LABEL%.
  exit /b 1
)
if /I not "%ACTUAL_SHA%"=="%EXPECTED_SHA%" (
  echo Integrity check failed before execution.
  echo Asset: %VERIFY_LABEL%
  echo Expected SHA256: %EXPECTED_SHA%
  echo Actual SHA256:   %ACTUAL_SHA%
  echo No files were installed or modified.
  exit /b 1
)
echo Verified %VERIFY_LABEL% SHA256.
exit /b 0

:emit_dry_run_json
set JSON_EXIT=%~1
set JSON_STATUS=%~2
set "JSON_INSTALL_DIR=%INSTALL_DIR:\=\\%"
set "JSON_ASSET_DIR="
if not "%ASSET_DIR%"=="" set "JSON_ASSET_DIR=%ASSET_DIR:\=\\%"
set JSON_INSTALLER_EXISTS=false
if exist "%INSTALLER_PS1%" set JSON_INSTALLER_EXISTS=true
echo {"schema":"easy_asr_bench.setup_dry_run.v1","mode":"dry_run","version":"%APP_VERSION%","status":"%JSON_STATUS%","exit_code":%JSON_EXIT%,"verify_release":%VERIFY_RELEASE%,"installer_exists":%JSON_INSTALLER_EXISTS%,"install_dir":"!JSON_INSTALL_DIR!","asset_dir":"!JSON_ASSET_DIR!","no_files_modified":true}
exit /b 0
