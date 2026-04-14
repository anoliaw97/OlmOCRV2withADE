# OlmOCRV2withADE - Detailed Windows Setup (Anaconda + llama.cpp)

This repository contains multiple apps. The current primary chat experience is:

- `scal_rebuild_webapp` (FastAPI + web UI)
- backend choices: `llama.cpp` (recommended) or `Ollama`

This guide is written for Windows + Anaconda Prompt and is intentionally detailed.

## 1) What You Need

- Windows machine (your remote desktop setup is supported)
- NVIDIA GPU (RTX A6000 works well)
- Anaconda or Miniconda
- Git
- CMake
- Visual Studio C++ Build Tools
- Conda env (example: `olmocr`)

Install prerequisites (admin terminal):

```bat
winget install Git.Git
winget install Kitware.CMake
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

## 2) Clone / Update Repository

```bat
cd /d "C:\Users\admin\Downloads"
git clone https://github.com/anoliaw97/OlmOCRV2withADE.git
cd /d "C:\Users\admin\Downloads\OlmOCRV2withADE"
git checkout rebuild/scal-openwebui-v1
git pull
```

If already cloned:

```bat
cd /d "C:\Users\admin\Downloads\olmocr-main-enhance2"
git checkout rebuild/scal-openwebui-v1
git pull
```

## 3) Create / Activate Conda Environment

If env already exists:

```bat
conda activate olmocr
```

If env does not exist:

```bat
conda create -n olmocr python=3.11 -y
conda activate olmocr
```

Install Python dependencies:

```bat
pip install -r requirements_olmocr_full.txt
pip install -e scal_webapp
```

## 4) Install CUDA Toolkit in Conda (for llama.cpp GPU build)

Important: `nvcc` alone is not enough; `cudart.lib` dev libraries are also needed.

```bat
conda activate olmocr
conda install -c nvidia cuda-toolkit=12.6 cuda-nvcc cuda-cudart-dev cuda-libraries-dev -y
```

Verify:

```bat
where nvcc
nvcc --version
dir "%CONDA_PREFIX%\Library\lib\cudart.lib"
```

## 5) Build llama.cpp (CUDA ON)

From repo root:

```bat
conda activate olmocr
set LLAMA_ENABLE_CUDA=ON
setup_llama_cpp_windows.bat
```

Expected output should show successful configure/build and a `llama-server.exe` path.

Default expected exe path:

```text
<repo>\llama.cpp\build\bin\Release\llama-server.exe
```

## 6) Model Location and Default GGUF

Recommended model folder:

```text
D:\models
```

Default configured model:

- repo: `bartowski/Qwen2.5-32B-Instruct-GGUF`
- file: `Qwen2.5-32B-Instruct-Q4_K_M.gguf`
- target path: `D:\models\Qwen2.5-32B-Instruct-Q4_K_M.gguf`

You can download from the web app UI (Advanced -> llama.cpp -> `Download Default GGUF`).

## 7) Run the App (Recommended Script)

One command starts `llama.cpp` and web app together:

```bat
cd /d "C:\Users\admin\Downloads\olmocr-main-enhance2"
conda activate olmocr
call run_scal_rebuild_stack_conda.bat olmocr
```

URLs:

- web app: `http://127.0.0.1:8092`
- llama.cpp server: `http://127.0.0.1:8081`

## 8) Run Components Manually (Optional)

Start llama.cpp only:

```bat
cd /d "C:\Users\admin\Downloads\olmocr-main-enhance2"
conda activate olmocr
set LLAMA_MODEL_PATH=D:\models\Qwen2.5-32B-Instruct-Q4_K_M.gguf
call run_llama_cpp_server_conda.bat olmocr
```

Start web app only (new terminal):

```bat
cd /d "C:\Users\admin\Downloads\olmocr-main-enhance2"
conda activate olmocr
python -m uvicorn scal_rebuild_webapp.main:app --host 127.0.0.1 --port 8000 --reload
```

If using this manual uvicorn command, open `http://127.0.0.1:8000`.

## 9) First-Time UI Configuration

1. Open web app
2. Switch to `Advanced` mode
3. Backend = `llama.cpp server`
4. Set:
   - `llama-server.exe` path
   - model dir: `D:\models`
   - model path: `D:\models\Qwen2.5-32B-Instruct-Q4_K_M.gguf`
5. Recommended performance settings:
   - `Ctx`: `8192`
   - `GPU Layers`: `999`
   - `Threads`: `12` (adjust to your CPU)
   - `flash-attn`: off first; enable after stable run
6. Click `Save llama.cpp`
7. Click `Load`

## 10) Performance Guidance (RTX A6000)

If RAM usage is too high or responses are slow:

- Keep `Ctx` at `8192` first (16384 uses much more memory)
- Keep `GPU Layers` high (`999`) to maximize offload
- Ensure CUDA build succeeded (no CPU-only fallback)
- Check GPU usage during generation:

```bat
nvidia-smi
```

- Check llama log:

```text
<repo>\scal_runtime_logs\llama_cpp_server.log
```

## 11) Common Errors and Fixes

### A) `CUDA Toolkit not found` / `Could NOT find CUDAToolkit`

Install dev libs in conda:

```bat
conda install -c nvidia cuda-cudart-dev cuda-libraries-dev -y
```

Then rerun:

```bat
set LLAMA_ENABLE_CUDA=ON
setup_llama_cpp_windows.bat
```

### B) `No connection ... 127.0.0.1:8081`

Means llama server is not up yet.

- Confirm model path exists
- Confirm exe path exists
- Click `Load` again after saving settings

### C) High memory even after closing prompt

Background process may still run:

```bat
tasklist | findstr /I "llama-server python ollama"
taskkill /F /IM llama-server.exe
taskkill /F /IM python.exe
```

## 12) Git Update Workflow (Existing Install)

```bat
cd /d "C:\Users\admin\Downloads\olmocr-main-enhance2"
conda activate olmocr
git fetch origin
git checkout rebuild/scal-openwebui-v1
git pull
```

If scripts or build settings changed, rerun:

```bat
set LLAMA_ENABLE_CUDA=ON
setup_llama_cpp_windows.bat
```

## 13) Backend Options

- `llama.cpp`: preferred local high-control backend
- `Ollama`: supported fallback backend in UI

LM Studio backend is intentionally not implemented in this branch.
