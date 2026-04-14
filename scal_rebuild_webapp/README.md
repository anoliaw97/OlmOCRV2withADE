# SCAL Rebuild WebApp (Custom)

This is a clean reconstruction branch/folder inspired by Open WebUI interaction patterns while keeping your local/offline SCAL workflow.

## What is included

- Supported backends:
  - `llama.cpp` server (`http://127.0.0.1:8081` by default)
  - Ollama (`http://127.0.0.1:11434`) for local model download/serving
- Streaming chat (`/api/chat/stream`) with token-by-token UI updates
- No forced document selection for chat
- PDF/JSON/HTML page preview available in both Layman and Advanced modes
- Resizable preview/chat/log panels and PDF pop-out action
- Session-based chat history
- Persisted TF-IDF index for extracted JSON/MD (load existing index; rebuild on demand)
- Assistant label uses active model name
- Layman / Advanced mode toggle
- Advanced mode enables power-user controls (logs, retrieval scope/type, model controls)
- Folder browse API for path selection (no manual copy/paste required)
- Ollama-backed model loader flow (pull on demand + auto-pull when switching missing models)
- Table export from retrieved results to Word/Excel-compatible files
- Session compaction and context-limit awareness via LangChain

## Run

For full step-by-step Windows + Anaconda + CUDA + llama.cpp setup, use the repository root guide:

- `README.md`

Quick run:

Use one command to launch `llama.cpp` + rebuilt UI:

`run_scal_rebuild_stack_conda.bat`

The rebuilt UI runs at:

`http://127.0.0.1:8092`

If using `llama.cpp`, ensure `llama-server` is available and the `LLAMA_SERVER_EXE` and `LLAMA_MODEL_PATH` environment variables point to valid paths if you are not using the script defaults.

If using Ollama backend, ensure Ollama is running locally.

## Notes

- Data root is persisted in `scal_rebuild_settings.json` (and can be overridden via `SCAL_DATA_ROOT`).
- Ollama URL defaults to `http://127.0.0.1:11434` and can be overridden via `SCAL_OLLAMA_BASE_URL`.
- llama.cpp URL defaults to `http://127.0.0.1:8081` and can be overridden via `SCAL_LLAMACPP_BASE_URL`.

## Performance tips (RTX A6000)

- Build `llama.cpp` with CUDA enabled (`setup_llama_cpp_windows.bat` now defaults to `GGML_CUDA=ON`).
- Start with `ctx=8192` instead of `16384` to reduce memory pressure and improve token speed.
- Use `gpu_layers=999` to offload as much as possible to GPU.
- If memory is still tight, switch to a smaller GGUF (for example 14B) for faster interactive chat.
