@echo off
setlocal

REM Usage:
REM   run_scal_webapp_conda.bat
REM   run_scal_webapp_conda.bat my_env_name

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=olmocr"

REM Run from repo root if launched elsewhere
cd /d "%~dp0"

REM Resolve conda activation script
set "CONDA_BAT="

REM 1) If already in a conda-enabled shell
if defined CONDA_EXE (
  for %%P in ("%CONDA_EXE%") do set "_CONDA_DIR=%%~dpP"
  if exist "%_CONDA_DIR%..\condabin\conda.bat" set "CONDA_BAT=%_CONDA_DIR%..\condabin\conda.bat"
  if exist "%_CONDA_DIR%conda.bat" set "CONDA_BAT=%_CONDA_DIR%conda.bat"
)

REM 2) Common install locations
if not defined CONDA_BAT if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\Miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\Miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\Anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\Anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\ProgramData\miniconda3\condabin\conda.bat" set "CONDA_BAT=C:\ProgramData\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\ProgramData\anaconda3\condabin\conda.bat" set "CONDA_BAT=C:\ProgramData\anaconda3\condabin\conda.bat"

REM 3) Fallback: discover via PATH (where conda)
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
  echo Could not find conda.bat automatically.
  echo.
  echo Checked common paths under your profile and ProgramData,
  echo and searched PATH using ^"where conda^".
  echo.
  echo Please run this from Anaconda Prompt, or edit this BAT and set CONDA_BAT manually.
  pause
  exit /b 1
)

set "USE_CONDA_RUN=0"
echo Activating Conda environment: %ENV_NAME%
call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
  echo Failed to activate Conda environment: %ENV_NAME%
  echo Falling back to conda run mode (no activate).
  set "USE_CONDA_RUN=1"
)

echo Installing/repairing required packages (missing packages will be added)...
if "%USE_CONDA_RUN%"=="1" (
  call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install --upgrade pip setuptools wheel >nul
  call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install -r requirements_olmocr_full.txt
) else (
  python -m pip install --upgrade pip setuptools wheel >nul
  python -m pip install -r requirements_olmocr_full.txt
)
if errorlevel 1 (
  echo Failed installing requirements.
  echo Try manually: pip install -r requirements_olmocr_full.txt
  pause
  exit /b 1
)

echo Ensuring CUDA-enabled PyTorch for local LLM...
if "%USE_CONDA_RUN%"=="1" (
  call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
  call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
) else (
  python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
)
if errorlevel 1 (
  echo Failed installing CUDA PyTorch.
  echo Local LLM may not work until torch CUDA install succeeds.
)

if "%USE_CONDA_RUN%"=="1" (
  call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install -e "scal_webapp"
) else (
  python -m pip install -e "scal_webapp"
)
if errorlevel 1 (
  echo Failed installing local scal_webapp package.
  pause
  exit /b 1
)

echo Starting SCAL Extraction + Offline RAG web app...
echo Open: http://localhost:8080
echo (Press Ctrl+C to stop)
if "%USE_CONDA_RUN%"=="1" (
  call "%CONDA_BAT%" run -n "%ENV_NAME%" scal-webapp
) else (
  scal-webapp
)
if errorlevel 1 (
  echo.
  echo Web app failed to start.
  echo If dependencies are missing, run:
  echo   pip install -r requirements_olmocr_full.txt
  echo   pip install -e scal_webapp
  pause
)

endlocal
