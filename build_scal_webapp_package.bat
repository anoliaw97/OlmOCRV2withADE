@echo off
setlocal

cd /d "%~dp0"

echo Building wheel package for SCAL webapp...
python -m pip install --upgrade pip build
if errorlevel 1 (
  echo Failed to install build tool.
  exit /b 1
)

python -m build "scal_webapp"
if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

echo.
echo Build complete. Package files are in:
echo   scal_webapp\dist
echo.
dir /b "scal_webapp\dist"

endlocal
