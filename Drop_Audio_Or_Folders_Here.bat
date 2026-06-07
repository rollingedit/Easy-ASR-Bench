@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Runtime not found. Running setup repair...
  call "%~dp0setup.bat" --local --no-post-setup-menu
  if errorlevel 1 (
    echo Repair failed. Run setup.bat, or see Logs\setup.log.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m app.main --interactive %*
pause
