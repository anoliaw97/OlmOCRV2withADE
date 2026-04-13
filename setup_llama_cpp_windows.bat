@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"
set "REPO_DIR=%CD%"
set "LLAMA_DIR=%REPO_DIR%\llama.cpp"
set "BUILD_DIR=%LLAMA_DIR%\build"

where git >nul 2>&1
if errorlevel 1 (
  echo ERROR: git is not installed or not on PATH.
  pause
  exit /b 1
)

where cmake >nul 2>&1
if errorlevel 1 (
  echo ERROR: cmake is not installed or not on PATH.
  echo Install CMake first, then rerun this script.
  pause
  exit /b 1
)

if exist "%LLAMA_DIR%\.git" (
  echo Updating existing llama.cpp checkout...
  git -C "%LLAMA_DIR%" pull --ff-only
) else (
  echo Cloning llama.cpp...
  git clone https://github.com/ggml-org/llama.cpp "%LLAMA_DIR%"
)

if errorlevel 1 (
  echo ERROR: Failed to fetch llama.cpp sources.
  pause
  exit /b 1
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo Configuring llama.cpp with CMake...
cmake -S "%LLAMA_DIR%" -B "%BUILD_DIR%" -DGGML_CUDA=OFF
if errorlevel 1 (
  echo ERROR: CMake configure failed.
  pause
  exit /b 1
)

echo Building llama.cpp...
cmake --build "%BUILD_DIR%" --config Release -j 8
if errorlevel 1 (
  echo ERROR: Build failed.
  pause
  exit /b 1
)

if exist "%BUILD_DIR%\bin\Release\llama-server.exe" (
  echo Build complete.
  echo llama-server.exe:
  echo   %BUILD_DIR%\bin\Release\llama-server.exe
) else (
  echo Build finished, but llama-server.exe was not found in the expected path.
  echo Check the build output under:
  echo   %BUILD_DIR%
)

endlocal
