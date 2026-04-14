@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"
set "REPO_DIR=%CD%"
set "LLAMA_DIR=%REPO_DIR%\llama.cpp"
set "BUILD_DIR=%LLAMA_DIR%\build"
set "CMAKE_EXE=cmake"

where git >nul 2>&1
if errorlevel 1 (
  echo ERROR: git is not installed or not on PATH.
  pause
  exit /b 1
)

where cmake >nul 2>&1
if errorlevel 1 (
  if exist "C:\Program Files\CMake\bin\cmake.exe" set "CMAKE_EXE=C:\Program Files\CMake\bin\cmake.exe"
  if exist "C:\Program Files (x86)\CMake\bin\cmake.exe" set "CMAKE_EXE=C:\Program Files (x86)\CMake\bin\cmake.exe"
)

if not exist "%CMAKE_EXE%" if /I not "%CMAKE_EXE%"=="cmake" (
  echo Using CMake from:
  echo   %CMAKE_EXE%
)

if /I "%CMAKE_EXE%"=="cmake" (
  where cmake >nul 2>&1
  if errorlevel 1 (
    echo ERROR: cmake is not installed or not on PATH.
    echo Install CMake first, then rerun this script.
    echo If CMake is already installed, reopen Anaconda Prompt after installation.
    pause
    exit /b 1
  )
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

if "%LLAMA_ENABLE_CUDA%"=="" set "LLAMA_ENABLE_CUDA=ON"
echo Build option: GGML_CUDA=%LLAMA_ENABLE_CUDA%

set "CMAKE_EXTRA_ARGS="
if /I "%LLAMA_ENABLE_CUDA%"=="ON" (
  if "%CUDAToolkit_ROOT%"=="" (
    if not "%CONDA_PREFIX%"=="" (
      if exist "%CONDA_PREFIX%\Library\lib\cudart.lib" (
        set "CUDAToolkit_ROOT=%CONDA_PREFIX%\Library"
      )
    )
  )
  if "%CUDAToolkit_ROOT%"=="" (
    if not "%CUDA_PATH%"=="" (
      set "CUDAToolkit_ROOT=%CUDA_PATH%"
    )
  )
  if "%CUDAToolkit_ROOT%"=="" (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" set "CUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    if "%CUDAToolkit_ROOT%"=="" if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5" set "CUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
    if "%CUDAToolkit_ROOT%"=="" if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4" set "CUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
  )

  if not "%CONDA_PREFIX%"=="" (
    if exist "%CONDA_PREFIX%\Library\bin\nvcc.exe" (
      set "PATH=%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\bin;%PATH%"
    )
  )

  if not "%CUDAToolkit_ROOT%"=="" (
    echo Using CUDAToolkit_ROOT=%CUDAToolkit_ROOT%
    set "CMAKE_EXTRA_ARGS=-DCUDAToolkit_ROOT=%CUDAToolkit_ROOT%"
  ) else (
    if not "%CONDA_PREFIX%"=="" (
      if not exist "%CONDA_PREFIX%\Library\lib\cudart.lib" (
        echo WARNING: cudart.lib not found in conda env: %CONDA_PREFIX%\Library\lib\cudart.lib
        echo Install CUDA dev libs with:
        echo   conda install -c nvidia cuda-cudart-dev cuda-libraries-dev -y
      )
    )
  )
)

if exist "%BUILD_DIR%\CMakeCache.txt" (
  echo Removing stale CMake cache...
  del /f /q "%BUILD_DIR%\CMakeCache.txt" >nul 2>&1
)
if exist "%BUILD_DIR%\CMakeFiles" (
  rmdir /s /q "%BUILD_DIR%\CMakeFiles" >nul 2>&1
)

echo Configuring llama.cpp with CMake...
"%CMAKE_EXE%" -S "%LLAMA_DIR%" -B "%BUILD_DIR%" -DGGML_CUDA=%LLAMA_ENABLE_CUDA% %CMAKE_EXTRA_ARGS%
if errorlevel 1 (
  echo ERROR: CMake configure failed.
  pause
  exit /b 1
)

echo Building llama.cpp...
"%CMAKE_EXE%" --build "%BUILD_DIR%" --config Release -j 8
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
