@echo off
setlocal

REM Usage:
REM   install_scal_webapp_package.bat
REM   install_scal_webapp_package.bat C:\path\to\scal_webapp_offline-0.1.0-py3-none-any.whl

set "WHEEL_PATH=%~1"

cd /d "%~dp0"

if "%WHEEL_PATH%"=="" (
  for %%F in ("scal_webapp\dist\scal_webapp_offline-*.whl") do set "WHEEL_PATH=%%~fF"
)

if "%WHEEL_PATH%"=="" (
  echo Wheel not found.
  echo Build first with:
  echo   build_scal_webapp_package.bat
  exit /b 1
)

echo Installing package:
echo   %WHEEL_PATH%

python -m pip install --upgrade "%WHEEL_PATH%"
if errorlevel 1 (
  echo Installation failed.
  exit /b 1
)

echo.
echo Installed successfully.
echo Launch with:
echo   scal-webapp

endlocal
