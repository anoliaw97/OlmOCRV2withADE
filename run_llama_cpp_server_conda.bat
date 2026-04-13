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
    echo Env %ENV_NAME% not available.
    pause
    exit /b 1
  )
  set "USE_CONDA_RUN=1"
)

if "%LLAMA_SERVER_EXE%"=="" set "LLAMA_SERVER_EXE=%REPO_DIR%\llama.cpp\build\bin\Release\llama-server.exe"
if "%LLAMA_MODEL_PATH%"=="" set "LLAMA_MODEL_PATH=%REPO_DIR%\models\model.gguf"
if "%LLAMA_CTX_SIZE%"=="" set "LLAMA_CTX_SIZE=16384"
if "%LLAMA_HOST%"=="" set "LLAMA_HOST=127.0.0.1"
if "%LLAMA_PORT%"=="" set "LLAMA_PORT=8081"

if not exist "%LLAMA_SERVER_EXE%" (
  echo ERROR: llama-server executable not found:
  echo   %LLAMA_SERVER_EXE%
  echo.
  echo Set LLAMA_SERVER_EXE to your llama-server.exe path and retry.
  pause
  exit /b 1
)

if not exist "%LLAMA_MODEL_PATH%" (
  echo ERROR: GGUF model file not found:
  echo   %LLAMA_MODEL_PATH%
  echo.
  echo Set LLAMA_MODEL_PATH to your .gguf file path and retry.
  pause
  exit /b 1
)

echo Starting llama.cpp server...
echo EXE : %LLAMA_SERVER_EXE%
echo MODEL: %LLAMA_MODEL_PATH%
echo URL  : http://%LLAMA_HOST%:%LLAMA_PORT%

if "%USE_CONDA_RUN%"=="1" (
  call "%CONDA_BAT%" run -n "%ENV_NAME%" "%LLAMA_SERVER_EXE%" -m "%LLAMA_MODEL_PATH%" --host %LLAMA_HOST% --port %LLAMA_PORT% -c %LLAMA_CTX_SIZE%
) else (
  "%LLAMA_SERVER_EXE%" -m "%LLAMA_MODEL_PATH%" --host %LLAMA_HOST% --port %LLAMA_PORT% -c %LLAMA_CTX_SIZE%
)

endlocal
