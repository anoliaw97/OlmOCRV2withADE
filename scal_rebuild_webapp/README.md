# SCAL Rebuild WebApp (Custom)

This is a clean reconstruction branch/folder inspired by Open WebUI interaction patterns while keeping your local/offline SCAL workflow.

## What is included

- Dual backend support:
  - standalone local inference API (`scal_inference_api`)
  - Ollama (`http://127.0.0.1:11434`) for local model download/serving
  - LocalAI local endpoint (`http://127.0.0.1:8080`)
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
- Optional multimodal pass (`use vision`) when page images (`_pageN.png/.jpg`) exist

## Run

Use one command to launch inference + rebuilt UI:

`run_scal_rebuild_stack_conda.bat`

The rebuilt UI runs at:

`http://127.0.0.1:8092`

If using Ollama backend, ensure Ollama is running locally.

## Notes

- Data root is persisted in `scal_rebuild_settings.json` (and can be overridden via `SCAL_DATA_ROOT`).
- Inference API URL defaults to `http://127.0.0.1:8010` and can be overridden via `SCAL_INFERENCE_API_URL`.
- Ollama URL defaults to `http://127.0.0.1:11434` and can be overridden via `SCAL_OLLAMA_BASE_URL`.
- LocalAI URL defaults to `http://127.0.0.1:8080` and can be overridden via `SCAL_LOCALAI_BASE_URL`.
