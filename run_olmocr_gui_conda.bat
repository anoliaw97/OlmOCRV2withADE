@echo off
setlocal EnableDelayedExpansion

REM Usage:
REM   run_olmocr_gui_conda.bat
REM   run_olmocr_gui_conda.bat my_env_name

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=olmocr"

cd /d "%~dp0"

echo [1/6] Detecting Conda...
set "CONDA_BAT="
if defined CONDA_EXE (
  for %%P in ("%CONDA_EXE%") do set "_CONDA_DIR=%%~dpP"
  if exist "%_CONDA_DIR%..\condabin\conda.bat" set "CONDA_BAT=%_CONDA_DIR%..\condabin\conda.bat"
  if exist "%_CONDA_DIR%conda.bat" set "CONDA_BAT=%_CONDA_DIR%conda.bat"
)
if not defined CONDA_BAT if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\ProgramData\miniconda3\condabin\conda.bat" set "CONDA_BAT=C:\ProgramData\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\ProgramData\anaconda3\condabin\conda.bat" set "CONDA_BAT=C:\ProgramData\anaconda3\condabin\conda.bat"

if not defined CONDA_BAT (
  echo ERROR: Could not find conda.bat automatically.
  echo Run from Anaconda Prompt or edit this BAT with your conda path.
  pause
  exit /b 1
)

echo [2/6] Ensuring env exists: %ENV_NAME%
call "%CONDA_BAT%" env list | findstr /R /I "^[ ]*%ENV_NAME%[ ]" >nul
if errorlevel 1 (
  echo Creating conda env: %ENV_NAME%
  call "%CONDA_BAT%" create -n "%ENV_NAME%" python=3.11 -y
  if errorlevel 1 (
    echo ERROR: Failed creating conda env.
    pause
    exit /b 1
  )
)

echo [3/6] Activating env: %ENV_NAME%
call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
  echo ERROR: Failed activating env.
  pause
  exit /b 1
)

echo [4/6] Installing/repairing dependencies (missing-only behavior via pip)
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_olmocr_full.txt
if errorlevel 1 (
  echo ERROR: Dependency installation failed.
  pause
  exit /b 1
)

echo [5/6] Installing Poppler helper (best effort)
call conda install -c conda-forge poppler -y >nul 2>&1

echo [6/6] Launching olmOCR GUI
echo Note: Models download only when you click Load VLM/Load LLM.
python olmocr_agentic_gui.py

endlocal
