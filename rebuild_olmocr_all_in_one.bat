@echo off
setlocal EnableDelayedExpansion

REM ================================================================
REM Full rebuild from scratch for environment: olmocr
REM - Reuses conda env if it exists (creates only if missing)
REM - Installs all required packages
REM - Installs local SCAL webapp package
REM - Does NOT pre-download models (GUI loads models on demand)
REM
REM Usage:
REM   rebuild_olmocr_all_in_one.bat
REM ================================================================

set "ENV_NAME=olmocr"
cd /d "%~dp0"

echo.
echo [1/8] Detecting Conda...

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
  for /f "delims=" %%I in ('where conda 2^>nul') do (
    if exist "%%~dpI..\condabin\conda.bat" (
      set "CONDA_BAT=%%~dpI..\condabin\conda.bat"
      goto :found_conda
    )
    if exist "%%~dpIconda.bat" (
      set "CONDA_BAT=%%~dpIconda.bat"
      goto :found_conda
    )
  )
)

:found_conda
if not defined CONDA_BAT (
  echo ERROR: Could not find conda.bat automatically.
  echo Run this script from Anaconda Prompt, or edit CONDA_BAT in this file.
  exit /b 1
)

echo Using: %CONDA_BAT%

echo.
echo [2/8] Checking environment: %ENV_NAME%
call "%CONDA_BAT%" env list | findstr /R /I "^[ ]*%ENV_NAME%[ ]" >nul
if errorlevel 1 (
  echo Environment not found. Creating: %ENV_NAME%
  call "%CONDA_BAT%" create -n "%ENV_NAME%" python=3.11 -y
  if errorlevel 1 (
    echo ERROR: Failed creating conda environment.
    exit /b 1
  )
) else (
  echo Environment exists. Reusing: %ENV_NAME%
)

echo.
echo [3/8] Activating environment: %ENV_NAME%
call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
  echo ERROR: Failed activating conda environment.
  exit /b 1
)

echo.
echo [4/8] Installing system helper dependencies (Poppler)
call conda install -c conda-forge poppler -y
if errorlevel 1 (
  echo WARNING: Poppler install failed. PDF rendering may fail until fixed.
)

echo.
echo [5/8] Installing Python dependencies
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo ERROR: pip/setuptools/wheel upgrade failed.
  exit /b 1
)

python -m pip install -r requirements_olmocr_full.txt
if errorlevel 1 (
  echo ERROR: Failed installing requirements_olmocr_full.txt
  exit /b 1
)

echo.
echo [6/8] Installing PyTorch CUDA build
python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
if errorlevel 1 (
  echo ERROR: PyTorch CUDA install failed.
  exit /b 1
)

echo.
echo [7/8] Installing local package
python -m pip install -e "scal_webapp"
if errorlevel 1 (
  echo ERROR: Failed installing local scal_webapp package.
  exit /b 1
)

echo.
echo [8/8] Final environment health check
python -m pip check

echo.
echo ================================================================
echo Rebuild complete.
echo Environment: %ENV_NAME%
echo.
echo Note: Models are NOT pre-downloaded by this script.
echo They will download automatically when you click Load VLM/Load LLM in GUI.
echo.
echo Launch options:
echo   1) SCAL web app:    scal-webapp
echo   2) olmOCR GUI:      python olmocr_agentic_gui.py
echo ================================================================

endlocal
exit /b 0
