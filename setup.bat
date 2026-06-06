@echo off
setlocal
cd /d "%~dp0"

set APP_NAME=Easy ASR Bench
set APP_VERSION=v0.2
set REPO_ZIP=https://github.com/rollingedit/Easy-ASR-Bench/archive/refs/tags/%APP_VERSION%.zip
set INSTALL_DIR=%LOCALAPPDATA%\Easy-ASR-Bench

if /I "%~1"=="--local" goto local_setup
if exist "%~dp0app\main.py" goto local_setup

echo Installing %APP_NAME%...
echo.
echo This setup file will download the app into:
echo %INSTALL_DIR%
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$install='%INSTALL_DIR%'; $logDir=Join-Path $install 'Logs'; New-Item -ItemType Directory -Force -Path $logDir | Out-Null; $log=Join-Path $logDir 'setup.log';" ^
  "'Starting setup at ' + (Get-Date) | Tee-Object -FilePath $log -Append;" ^
  "$zip=Join-Path $env:TEMP 'Easy-ASR-Bench-%APP_VERSION%.zip'; $extract=Join-Path $env:TEMP 'Easy-ASR-Bench-%APP_VERSION%';" ^
  "Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue;" ^
  "Remove-Item -LiteralPath $extract -Recurse -Force -ErrorAction SilentlyContinue;" ^
  "New-Item -ItemType Directory -Force -Path $install | Out-Null;" ^
  "Invoke-WebRequest -Uri '%REPO_ZIP%' -OutFile $zip; $hash=(Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash; 'Downloaded source archive SHA256: ' + $hash | Tee-Object -FilePath $log -Append;" ^
  "Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force;" ^
  "$src=Get-ChildItem -LiteralPath $extract -Directory | Select-Object -First 1;" ^
  "Copy-Item -Path (Join-Path $src.FullName '*') -Destination $install -Recurse -Force;" ^
  "Start-Process -FilePath (Join-Path $install 'setup.bat') -ArgumentList '--local' -Wait"

if errorlevel 1 (
  echo Setup failed.
  pause
  exit /b 1
)

echo.
echo Setup complete. Opening installed folder...
explorer "%INSTALL_DIR%"
pause
exit /b 0

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

echo Installing Python packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
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

echo Running self-test...
".venv\Scripts\python.exe" -m app.self_test --config config.json
if errorlevel 1 (
  pause
  exit /b 1
)

echo.
echo Setup complete. Drop models into Models, then use Run.bat.
pause
