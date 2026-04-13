@echo off
setlocal EnableDelayedExpansion

REM One-command launcher:
REM 1) llama.cpp server on 8081 (new window)
REM 2) Rebuild UI on 8092 (current window)

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=olmocr"

cd /d "%~dp0"
set "REPO_DIR=%CD%"

set "LLAMA_RUNNER=%TEMP%\scal_llama_runner_%ENV_NAME%.bat"

(
  echo @echo off
  echo cd /d "%REPO_DIR%"
  echo call "%~dp0run_llama_cpp_server_conda.bat" %ENV_NAME%
) > "%LLAMA_RUNNER%"

echo =========================================================
echo SCAL one-command launcher (llama.cpp + rebuild)
echo Env: %ENV_NAME%
echo =========================================================

echo Starting llama.cpp server in separate window...
start "SCAL llama.cpp Server" "%LLAMA_RUNNER%"

echo llama.cpp server launched in separate window ^(startup continues in background^).
echo Starting rebuild web app in current window...
call "%~dp0run_scal_rebuild_webapp_conda.bat" %ENV_NAME%

endlocal
