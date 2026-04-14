# OCRLLMVLMMLPETROAFRO Desktop

Tauri 2 desktop app with a SolidJS frontend and Bun workspace tooling.

This project now includes:

- OCR ingestion runtime command (`ingest_ocr`)
- local `llama.cpp` runtime command (`run_llama_cpp`)
- integrated UI flow for OCR context + local model prompting

## Architecture

```text
.
|- package.json
|- turbo.json
|- tsconfig.base.json
`- packages/
   |- app/               # Solid + Vite frontend
   |- core/              # shared TS types/contracts
   `- desktop/           # Tauri app + Rust runtime modules
      `- src-tauri/
         |- src/ocr.rs   # OCR ingestion module
         |- src/llama.rs # llama.cpp runtime module
         `- src/lib.rs   # Tauri command bridge
```

## Prerequisites (Windows)

Install these first:

1. Bun (1.2+)
2. Rust toolchain via `rustup`
3. Visual Studio C++ Build Tools (Desktop development with C++)
4. WebView2 Runtime (usually already present on Windows 11)
5. Optional but recommended: GitHub CLI (`gh`) for PR flow

Useful links:

- https://bun.sh
- https://www.rust-lang.org/tools/install
- https://tauri.app/start/prerequisites/
- https://github.com/ggerganov/llama.cpp
- https://github.com/tesseract-ocr/tesseract

## External Runtime Dependencies

### 1) Tesseract OCR

`ingest_ocr` can:

- read plain text files directly (`.txt`, `.md`, `.csv`, `.json`, etc.)
- run Tesseract for image/PDF OCR input

Install Tesseract and ensure either:

- `tesseract.exe` is on `PATH`, or
- you provide an absolute executable path in the UI (`Tesseract Path`)

Example path:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 2) llama.cpp

`run_llama_cpp` expects:

- `llama-cli.exe` available on `PATH`, or explicit path in UI
- a local `.gguf` model file path

Build `llama.cpp` on Windows (PowerShell example):

```powershell
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -S . -B build -DLLAMA_CURL=OFF
cmake --build build --config Release
```

Typical executable location:

```text
<llama.cpp>\build\bin\Release\llama-cli.exe
```

## Clone and Install

```powershell
git clone https://github.com/anoliaw97/OlmOCRV2withADE.git
cd OlmOCRV2withADE
bun install
```

If `bun` is not recognized in current shell after install, restart terminal.

## Run in Development

### Recommended (desktop + frontend together)

```powershell
bun run dev:desktop
```

This starts:

- Vite frontend at `http://localhost:1420`
- Tauri desktop shell that loads the frontend

### Frontend only

```powershell
bun run dev:app
```

## Verify Setup

Type checks:

```powershell
bun run typecheck
```

Build frontend:

```powershell
bun run build:app
```

Build desktop package:

```powershell
bun run build:desktop
```

## How to Use OCR + llama.cpp in the App

1. Start app with `bun run dev:desktop`
2. In left panel, set:
   - `OCR Input Path` (image/PDF/text file)
   - `OCR Language` (default `eng`)
   - `Tesseract Path` (optional if not on PATH)
3. Click **Ingest OCR**
4. In right panel, set:
   - `llama.cpp Model Path` (local `.gguf`)
   - `llama-cli Path` (optional if on PATH)
   - `System Prompt`
5. Enter a prompt in composer and click **Run llama.cpp**

The OCR text is passed as context into the local model runtime command.

## Command Reference

Workspace scripts:

- `bun run dev` -> alias to `dev:desktop`
- `bun run dev:desktop`
- `bun run dev:app`
- `bun run build`
- `bun run build:app`
- `bun run build:desktop`
- `bun run typecheck`

## Troubleshooting

### `bun` not recognized

- reopen terminal after Bun install
- verify: `bun --version`

### Tauri fails with icon error

- ensure `packages/desktop/src-tauri/icons/icon.ico` exists

### `Could not find tesseract executable`

- install Tesseract or set absolute `Tesseract Path` in UI
- verify from terminal: `tesseract --version`

### `Could not find llama-cli executable`

- build/download `llama.cpp`
- set `llama-cli` path in UI or add folder to `PATH`
- verify from terminal: `llama-cli --help`

### Model not found

- verify `llama.cpp Model Path` points to an existing `.gguf`

### Web server not on port 1420

- ensure no other service is using port 1420

## Current Status

Implemented:

- Tauri command bridge (`app_info`, `run_pipeline`, `ingest_ocr`, `run_llama_cpp`)
- OCR ingestion runtime module with Tesseract fallback
- Local `llama.cpp` runtime invocation module
- Solid desktop UI wiring for OCR + local inference

Next suggested enhancements:

- file picker integration via Tauri dialog plugin
- response streaming/token-level updates from llama runtime
- persistent runtime settings (paths, prompt templates, model profiles)
