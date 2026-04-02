@echo off
setlocal EnableDelayedExpansion

REM One-command launcher (no classic app):
REM 1) Inference API on 8010 (new window)
REM 2) Rebuild UI on 8092 (current window)

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=olmocr"

cd /d "%~dp0"
set "REPO_DIR=%CD%"

set "INFER_RUNNER=%TEMP%\scal_infer_runner_%ENV_NAME%.bat"

(
  echo @echo off
  echo cd /d "%REPO_DIR%"
  echo call "%~dp0run_scal_inference_api_conda.bat" %ENV_NAME%
) > "%INFER_RUNNER%"

echo =========================================================
echo SCAL one-command launcher (inference + rebuild)
echo Env: %ENV_NAME%
echo =========================================================

echo Starting inference API in separate window...
start "SCAL Inference API" "%INFER_RUNNER%"

echo Waiting for inference health on http://127.0.0.1:8010/v1/health ...
set "INFER_OK=0"
for /L %%I in (1,1,240) do (
  powershell -NoProfile -Command "try { Invoke-WebRequest 'http://127.0.0.1:8010/v1/health' -UseBasicParsing -TimeoutSec 2 ^| Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
  if not errorlevel 1 (
    set "INFER_OK=1"
    goto :infer_ready
  )
  if %%I==60 echo Still waiting for inference API ^(60s^)...
  if %%I==120 echo Still waiting for inference API ^(120s^)...
  if %%I==180 echo Still waiting for inference API ^(180s^)...
  timeout /t 1 /nobreak >nul
)

:infer_ready
if "%INFER_OK%"=="0" (
  echo.
  echo ERROR: Inference API did not become healthy on 8010.
  echo Check the "SCAL Inference API" window for traceback and fix that first.
  pause
  exit /b 1
)

echo Inference API is healthy.
echo Starting rebuild web app in current window...
call "%~dp0run_scal_rebuild_webapp_conda.bat" %ENV_NAME%

endlocal
