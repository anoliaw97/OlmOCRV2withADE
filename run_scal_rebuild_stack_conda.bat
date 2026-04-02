@echo off
setlocal

REM One-command launcher for full local stack:
REM 1) inference API (8010)
REM 2) classic SCAL webapp (8080)
REM 3) rebuild webapp (8092)

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=olmocr"

cd /d "%~dp0"
set "REPO_DIR=%CD%"

set "CONDA_BAT="
if defined CONDA_EXE (
  for %%P in ("%CONDA_EXE%") do set "_CONDA_DIR=%%~dpP"
  if exist "%_CONDA_DIR%..\condabin\conda.bat" set "CONDA_BAT=%_CONDA_DIR%..\condabin\conda.bat"
  if exist "%_CONDA_DIR%conda.bat" set "CONDA_BAT=%_CONDA_DIR%conda.bat"
)
if not defined CONDA_BAT if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\Miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\Miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\Anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\Anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\ProgramData\miniconda3\condabin\conda.bat" set "CONDA_BAT=C:\ProgramData\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\ProgramData\anaconda3\condabin\conda.bat" set "CONDA_BAT=C:\ProgramData\anaconda3\condabin\conda.bat"

if not defined CONDA_BAT (
  echo ERROR: Could not find conda.bat.
  echo Please run from Anaconda Prompt or edit this script with your conda path.
  pause
  exit /b 1
)

echo =========================================================
echo SCAL one-command launcher
echo Env: %ENV_NAME%
echo =========================================================

echo Checking Conda environment...
call "%CONDA_BAT%" run -n "%ENV_NAME%" python --version >nul 2>&1
if errorlevel 1 (
  echo Creating environment: %ENV_NAME%
  call "%CONDA_BAT%" create -n "%ENV_NAME%" python=3.11 -y
  if errorlevel 1 (
    echo Failed to create env %ENV_NAME%.
    pause
    exit /b 1
  )
)

echo Installing/updating shared dependencies ^(one-time sequential setup^)...
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install --upgrade pip wheel "setuptools<82"
if errorlevel 1 goto :deps_fail

call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install -r requirements_olmocr_full.txt
if errorlevel 1 goto :deps_fail

echo Installing CUDA PyTorch stack ^(aligned versions^) ...
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
if errorlevel 1 (
  echo WARNING: CUDA PyTorch install failed. Services may run CPU-only or fail for VLM.
)

call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install "setuptools<82"

set "INFER_RUNNER=%TEMP%\scal_infer_runner_%ENV_NAME%.bat"
set "CLASSIC_RUNNER=%TEMP%\scal_classic_runner_%ENV_NAME%.bat"

(
  echo @echo off
  echo cd /d "%REPO_DIR%"
  echo set "HF_HUB_DISABLE_XET=1"
  echo set "HF_HUB_ENABLE_HF_TRANSFER=1"
  echo set "HF_HUB_DOWNLOAD_TIMEOUT=120"
  echo set "PYTHONPATH=%REPO_DIR%;%%PYTHONPATH%%"
  echo call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m uvicorn scal_inference_api.main:app --host 127.0.0.1 --port 8010 --app-dir "%REPO_DIR%"
  echo set "EC=%%ERRORLEVEL%%"
  echo if not "%%EC%%"=="0" ^(
  echo   echo.
  echo   echo Inference API failed ^(error %%EC%%^).
  echo   echo Check Python and dependency logs above.
  echo   pause
  echo ^)
) > "%INFER_RUNNER%"

(
  echo @echo off
  echo cd /d "%REPO_DIR%"
  echo set "PYTHONPATH=%REPO_DIR%;%%PYTHONPATH%%"
  echo call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m uvicorn scal_webapp.backend.main:app --host 127.0.0.1 --port 8080 --app-dir "%REPO_DIR%"
  echo set "EC=%%ERRORLEVEL%%"
  echo if not "%%EC%%"=="0" ^(
  echo   echo.
  echo   echo Classic UI failed ^(error %%EC%%^).
  echo   echo Check Python and dependency logs above.
  echo   pause
  echo ^)
) > "%CLASSIC_RUNNER%"

echo Starting services in separate windows...
start "SCAL Inference API" "%INFER_RUNNER%"

timeout /t 2 /nobreak >nul

set "INFER_OK=0"
for /L %%I in (1,1,20) do (
  powershell -NoProfile -Command "try { Invoke-WebRequest 'http://127.0.0.1:8010/v1/health' -UseBasicParsing -TimeoutSec 2 ^| Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
  if not errorlevel 1 (
    set "INFER_OK=1"
    goto :infer_ready
  )
  timeout /t 1 /nobreak >nul
)

:infer_ready
if "%INFER_OK%"=="0" (
  echo WARNING: Inference API did not become healthy on 8010. Check the "SCAL Inference API" window.
) else (
  echo Inference API is healthy on 8010.
)

start "SCAL Classic UI" "%CLASSIC_RUNNER%"

timeout /t 3 /nobreak >nul

echo Starting rebuild webapp in this window...
set "SCAL_INFERENCE_API_URL=http://127.0.0.1:8010"
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m uvicorn scal_rebuild_webapp.main:app --host 127.0.0.1 --port 8092 --app-dir "%REPO_DIR%"
goto :eof

:deps_fail
echo.
echo Dependency installation failed.
echo Try re-running from Anaconda Prompt:
echo   run_scal_rebuild_stack_conda.bat %ENV_NAME%
pause
exit /b 1
