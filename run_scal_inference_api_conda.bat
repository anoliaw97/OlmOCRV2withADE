@echo off
setlocal EnableDelayedExpansion

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=olmocr"

cd /d "%~dp0"

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

echo Activating env: %ENV_NAME%
call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
  echo Creating env: %ENV_NAME%
  call "%CONDA_BAT%" create -n "%ENV_NAME%" python=3.11 -y
  call "%CONDA_BAT%" activate "%ENV_NAME%"
)

echo Installing/updating dependencies...
python -m pip install --upgrade pip setuptools wheel
python -m pip install fastapi uvicorn[standard] pydantic transformers torch hf_transfer

echo Configuring Hugging Face download settings...
set "HF_HUB_DISABLE_XET=1"
set "HF_HUB_ENABLE_HF_TRANSFER=1"
set "HF_HUB_DOWNLOAD_TIMEOUT=120"

echo Starting local inference API...
echo Open http://127.0.0.1:8010/v1/health
python -m uvicorn scal_inference_api.main:app --host 127.0.0.1 --port 8010 --workers 1

endlocal
