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

echo Inference API launched in separate window ^(startup continues in background^).
echo Starting rebuild web app in current window...
call "%~dp0run_scal_rebuild_webapp_conda.bat" %ENV_NAME%

endlocal
