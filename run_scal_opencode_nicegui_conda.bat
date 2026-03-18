@echo off
setlocal EnableDelayedExpansion

REM Usage:
REM   run_scal_opencode_nicegui_conda.bat
REM   run_scal_opencode_nicegui_conda.bat olmocr

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
python -m pip install -r requirements_olmocr_full.txt
python -m pip install nicegui scikit-learn joblib pypdf python-docx pandas openpyxl beautifulsoup4

echo Launching NiceGUI app...
echo Open http://localhost:8088
python scal_opencode_nicegui.py

endlocal
