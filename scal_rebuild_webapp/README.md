# SCAL Rebuild WebApp (Custom)

This is a clean reconstruction branch/folder inspired by Open WebUI interaction patterns while keeping your local/offline SCAL workflow.

## What is included

- Dual backend support:
  - standalone local inference API (`scal_inference_api`)
  - Ollama (`http://127.0.0.1:11434`) for local model download/serving
- Streaming chat (`/api/chat/stream`) with token-by-token UI updates
- No forced document selection for chat
- Session-based chat history
- Auto-built TF-IDF index for extracted JSON/MD
- Assistant label uses active model name
- Layman / Advanced mode toggle
- Advanced mode includes previous UI quick link (`8090`)
- Folder browse API for path selection (no manual copy/paste required)

## Run

Use one command to launch both services:

`run_scal_rebuild_stack_conda.bat`

The rebuilt UI runs at:

`http://127.0.0.1:8092`

If using Ollama backend, ensure Ollama is running locally.

## Notes

- Data root is persisted in `scal_rebuild_settings.json` (and can be overridden via `SCAL_DATA_ROOT`).
- Inference API URL defaults to `http://127.0.0.1:8010` and can be overridden via `SCAL_INFERENCE_API_URL`.
- Ollama URL defaults to `http://127.0.0.1:11434` and can be overridden via `SCAL_OLLAMA_BASE_URL`.
