@echo off
setlocal EnableDelayedExpansion

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=olmocr"

cd /d "%~dp0"
set "REPO_DIR=%CD%"

set "CONDA_BAT="
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\ProgramData\miniconda3\condabin\conda.bat" set "CONDA_BAT=C:\ProgramData\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\ProgramData\anaconda3\condabin\conda.bat" set "CONDA_BAT=C:\ProgramData\anaconda3\condabin\conda.bat"

if not defined CONDA_BAT (
  echo ERROR: Could not find conda.bat.
  pause
  exit /b 1
)

set "USE_CONDA_RUN=0"
echo Activating env: %ENV_NAME%
call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
  echo Activate failed. Trying conda run mode...
  call "%CONDA_BAT%" run -n "%ENV_NAME%" python --version >nul 2>&1
  if errorlevel 1 (
    echo Env not ready. Creating env: %ENV_NAME%
    call "%CONDA_BAT%" create -n "%ENV_NAME%" python=3.11 -y
  )
  call "%CONDA_BAT%" run -n "%ENV_NAME%" python --version >nul 2>&1
  if errorlevel 1 (
    echo Failed to use env %ENV_NAME% with conda run.
    pause
    exit /b 1
  )
  set "USE_CONDA_RUN=1"
)

echo Installing/updating dependencies...
set "NEED_INSTALL=0"
if "%USE_CONDA_RUN%"=="1" (
  call "%CONDA_BAT%" run -n "%ENV_NAME%" python -c "import fastapi,uvicorn,pydantic,transformers,torch" >nul 2>&1
  if errorlevel 1 set "NEED_INSTALL=1"
) else (
  python -c "import fastapi,uvicorn,pydantic,transformers,torch" >nul 2>&1
  if errorlevel 1 set "NEED_INSTALL=1"
)

if "%NEED_INSTALL%"=="1" (
  echo Dependencies missing/incompatible. Installing...
  if "%USE_CONDA_RUN%"=="1" (
    call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install --upgrade pip wheel "setuptools<82"
    call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install fastapi uvicorn[standard] pydantic "transformers==4.57.3" hf_transfer
    call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip uninstall -y torchvision torchaudio >nul 2>&1
    call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
  ) else (
    python -m pip install --upgrade pip wheel "setuptools<82"
    python -m pip install fastapi uvicorn[standard] pydantic "transformers==4.57.3" hf_transfer
    python -m pip uninstall -y torchvision torchaudio >nul 2>&1
    python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
  )
) else (
  echo Dependencies already satisfied. Skipping pip install.
)

if "%USE_CONDA_RUN%"=="1" (
  call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip uninstall -y torchvision torchaudio >nul 2>&1
) else (
  python -m pip uninstall -y torchvision torchaudio >nul 2>&1
)

echo Configuring Hugging Face download settings...
set "HF_HUB_DISABLE_XET=1"
set "HF_HUB_ENABLE_HF_TRANSFER=1"
set "HF_HUB_DOWNLOAD_TIMEOUT=120"

echo Starting local inference API...
echo Open http://127.0.0.1:8010/v1/health
if "%USE_CONDA_RUN%"=="1" (
  call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m uvicorn scal_inference_api.main:app --host 127.0.0.1 --port 8010 --app-dir "%REPO_DIR%"
) else (
  set "PYTHONPATH=%REPO_DIR%;%PYTHONPATH%"
  python -m uvicorn scal_inference_api.main:app --host 127.0.0.1 --port 8010 --app-dir "%REPO_DIR%"
)

endlocal
