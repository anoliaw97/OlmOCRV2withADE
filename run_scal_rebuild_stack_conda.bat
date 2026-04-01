@echo off
setlocal

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=olmocr"

cd /d "%~dp0"

echo Starting SCAL inference API in a new window...
start "SCAL Inference API" cmd /k "call "%~dp0run_scal_inference_api_conda.bat" %ENV_NAME%"

echo Waiting 3 seconds before launching rebuild web app...
timeout /t 3 /nobreak >nul

echo Starting SCAL rebuild web app in current window...
call "%~dp0run_scal_rebuild_webapp_conda.bat" %ENV_NAME%

endlocal
