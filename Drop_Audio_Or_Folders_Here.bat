@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Runtime not found. Running setup repair...
  call "%~dp0setup.bat" --local
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m app.main --doctor
".venv\Scripts\python.exe" -m app.main --interactive %*
pause
