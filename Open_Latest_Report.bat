@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "OUTPUT_DIR=%~dp0Output"
if not exist "%OUTPUT_DIR%" (
  echo Output folder was not found.
  pause
  exit /b 1
)

set "LATEST="
for /f "delims=" %%D in ('dir /b /ad /o-d "%OUTPUT_DIR%" 2^>nul') do (
  if exist "%OUTPUT_DIR%\%%D\final_results.html" (
    set "LATEST=%OUTPUT_DIR%\%%D\final_results.html"
    goto open_report
  )
  if exist "%OUTPUT_DIR%\%%D\compare.html" (
    set "LATEST=%OUTPUT_DIR%\%%D\compare.html"
    goto open_report
  )
)

echo No final_results.html or compare.html report was found under Output.
echo Run a benchmark first, then use this launcher again.
pause
exit /b 1

:open_report
echo Opening latest HTML report:
echo   %LATEST%
start "" "%LATEST%"
exit /b 0
