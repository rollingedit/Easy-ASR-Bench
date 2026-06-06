@echo off
cd /d "%~dp0"
if /I "%~1"=="--doctor" goto doctor
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
exit /b 0

:doctor
if not exist ".venv\Scripts\python.exe" (
  if exist "app\main.py" (
    python -m app.main --doctor
    exit /b %errorlevel%
  ) else (
    echo Runtime not found. Run setup.bat first.
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m app.main --doctor
exit /b %errorlevel%
