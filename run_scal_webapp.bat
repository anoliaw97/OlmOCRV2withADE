@echo off
setlocal

REM Run from repo root if launched elsewhere
cd /d "%~dp0"

echo Starting SCAL Extraction + Offline RAG web app...
echo.

python -m scal_webapp
if errorlevel 1 (
  echo.
  echo Web app failed to start.
  echo If dependencies are missing, run:
  echo   pip install -r scal_webapp\requirements.txt
  pause
)

endlocal
