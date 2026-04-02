@echo off
setlocal

REM One-command launcher for full local stack:
REM 1) inference API (8010)
REM 2) classic SCAL webapp (8080)
REM 3) rebuild webapp (8092)

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=olmocr"

cd /d "%~dp0"

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
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :deps_fail

call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install -r requirements_olmocr_full.txt
if errorlevel 1 goto :deps_fail

echo Installing CUDA PyTorch stack ^(aligned versions^) ...
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
if errorlevel 1 (
  echo WARNING: CUDA PyTorch install failed. Services may run CPU-only or fail for VLM.
)

echo Installing classic webapp package...
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install -e "scal_webapp" --no-deps
if errorlevel 1 goto :deps_fail

echo Starting services in separate windows...
start "SCAL Inference API" cmd /k "set HF_HUB_DISABLE_XET=1 ^& set HF_HUB_ENABLE_HF_TRANSFER=1 ^& set HF_HUB_DOWNLOAD_TIMEOUT=120 ^& call ""%CONDA_BAT%"" run -n ""%ENV_NAME%"" python -m uvicorn scal_inference_api.main:app --host 127.0.0.1 --port 8010 --workers 1"

timeout /t 2 /nobreak >nul

start "SCAL Classic UI" cmd /k "call ""%CONDA_BAT%"" run -n ""%ENV_NAME%"" scal-webapp"

timeout /t 3 /nobreak >nul

echo Starting rebuild webapp in this window...
set "SCAL_INFERENCE_API_URL=http://127.0.0.1:8010"
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m uvicorn scal_rebuild_webapp.main:app --host 127.0.0.1 --port 8092 --workers 1
goto :eof

:deps_fail
echo.
echo Dependency installation failed.
echo Try re-running from Anaconda Prompt:
echo   run_scal_rebuild_stack_conda.bat %ENV_NAME%
pause
exit /b 1
