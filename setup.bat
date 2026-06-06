@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set APP_NAME=Easy ASR Bench
set APP_VERSION=v0.2.9
set INSTALL_DIR=%LOCALAPPDATA%\Easy-ASR-Bench
set INSTALLER_PS1=%~dp0installer\install.ps1
set INSTALLER_URL=https://raw.githubusercontent.com/rollingedit/Easy-ASR-Bench/%APP_VERSION%/installer/install.ps1

if /I "%~1"=="--dry-run" goto dry_run
if /I "%~2"=="--dry-run" goto dry_run
if /I "%~1"=="--doctor" goto doctor
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

powershell -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER_PS1%" ^
  -InstallDir "%INSTALL_DIR%" ^
  -Version "%APP_VERSION%"

if errorlevel 1 (
  echo Setup failed. Check %INSTALL_DIR%\Logs\setup.log
  pause
  exit /b 1
)

echo.
echo Setup complete. Opening installed folder...
explorer "%INSTALL_DIR%"
pause
exit /b 0

:dry_run
echo %APP_NAME% setup dry run
echo Version: %APP_VERSION%
echo Install folder: %INSTALL_DIR%
echo Mode: validates installer inputs without changing files.
if not exist "%INSTALLER_PS1%" (
  echo Installer script was not found beside setup.bat.
  echo Standalone setup will download it from:
  echo   %INSTALLER_URL%
  exit /b 0
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER_PS1%" ^
  -InstallDir "%INSTALL_DIR%" ^
  -Version "%APP_VERSION%" ^
  -DryRun
exit /b %errorlevel%

:doctor
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m app.doctor --config config.json
  exit /b %errorlevel%
)
if exist "%INSTALL_DIR%\.venv\Scripts\python.exe" (
  "%INSTALL_DIR%\.venv\Scripts\python.exe" -m app.doctor --config "%INSTALL_DIR%\config.json"
  exit /b %errorlevel%
)
echo Easy ASR Bench is not installed here and no installed runtime was found.
exit /b 1

:uninstall
if not exist "%INSTALLER_PS1%" (
  echo Installer script was not found next to setup.bat.
  echo Run setup.bat first, or uninstall from the installed folder.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER_PS1%" ^
  -InstallDir "%INSTALL_DIR%" ^
  -Uninstall
exit /b %errorlevel%

:local_setup
echo %APP_NAME% - local setup
echo.

set PYEXE=
py -3.11 -c "import sys" >nul 2>nul
if not errorlevel 1 set PYEXE=py -3.11

if "%PYEXE%"=="" (
  py -3.12 -c "import sys" >nul 2>nul
  if not errorlevel 1 set PYEXE=py -3.12
)

if "%PYEXE%"=="" (
  echo Python 3.11 or 3.12 was not found. Attempting install with winget...
  winget install -e --id Python.Python.3.11
  py -3.11 -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo Python install was not detected. Install Python 3.11 and rerun setup.bat.
    pause
    exit /b 1
  )
  set PYEXE=py -3.11
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

echo.
echo Setup complete. Drop models into Models, then use Run.bat.
pause
exit /b 0
