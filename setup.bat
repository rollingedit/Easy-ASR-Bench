@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set APP_NAME=Easy ASR Bench
set APP_VERSION=v0.3.9
set INSTALL_DIR=%LOCALAPPDATA%\Easy-ASR-Bench
set INSTALLER_PS1=%~dp0installer\install.ps1
set INSTALLER_URL=https://github.com/rollingedit/Easy-ASR-Bench/releases/download/%APP_VERSION%/install.ps1
set INSTALLER_SHA256=sha256:1c6ab7454d666f7f5e92bb40fa04764a5ad20827bda9fe305a7e8e5c72cd7ed2
set VERIFY_RELEASE=0
set ASSET_DIR=
set NEXT_IS_ASSET_DIR=0
set DOCTOR_ARGS=
set NO_POST_SETUP_MENU=0
set INSTALLER_MODE_ARGS=

for %%A in (%*) do (
  if "!NEXT_IS_ASSET_DIR!"=="1" (
    set "ASSET_DIR=%%~A"
    set NEXT_IS_ASSET_DIR=0
  ) else (
    if /I "%%~A"=="--verify-release" set VERIFY_RELEASE=1
    if /I "%%~A"=="--asset-dir" set NEXT_IS_ASSET_DIR=1
    if /I "%%~A"=="--json" set DOCTOR_ARGS=!DOCTOR_ARGS! --json
    if /I "%%~A"=="--strict" set DOCTOR_ARGS=!DOCTOR_ARGS! --strict
    if /I "%%~A"=="--repair-plan" set DOCTOR_ARGS=!DOCTOR_ARGS! --repair-plan
    if /I "%%~A"=="--repair-all-safe" set DOCTOR_ARGS=!DOCTOR_ARGS! --repair-all-safe
    if /I "%%~A"=="--validate-real-smoke" set DOCTOR_ARGS=!DOCTOR_ARGS! --validate-real-smoke
    if /I "%%~A"=="--install-deps" set DOCTOR_ARGS=!DOCTOR_ARGS! --install-deps
    if /I "%%~A"=="--allow-downloads" set DOCTOR_ARGS=!DOCTOR_ARGS! --allow-downloads
    if /I "%%~A"=="--no-network" set DOCTOR_ARGS=!DOCTOR_ARGS! --no-network
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
  set INSTALLER_PS1=%TEMP%\Easy-ASR-Bench-install-%APP_VERSION%.ps1
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri '%INSTALLER_URL%' -OutFile '%INSTALLER_PS1%'"
  if errorlevel 1 (
    echo Could not download installer script.
    pause
    exit /b 1
  )
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
    exit /b 0
  )
  echo Downloading installer script for release verification...
  set INSTALLER_PS1=%TEMP%\Easy-ASR-Bench-install-%APP_VERSION%.ps1
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri '%INSTALLER_URL%' -OutFile '%INSTALLER_PS1%'"
  if errorlevel 1 (
    echo Could not download installer script.
    exit /b 1
  )
)
call :verify_sha "%INSTALLER_PS1%" "%INSTALLER_SHA256%" "installer/install.ps1"
if errorlevel 1 exit /b 1
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
  exit /b !ERRORLEVEL!
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER_PS1%" ^
  -InstallDir "%INSTALL_DIR%" ^
  -Version "%APP_VERSION%" ^
  -DryRun ^
  %INSTALLER_MODE_ARGS%
exit /b !ERRORLEVEL!

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
  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
  py -3.12 -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo Python install was not detected. Install Python 3.12 and rerun setup.bat.
    pause
    exit /b 1
  )
  set PYEXE=py -3.12
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
