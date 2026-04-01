# SCAL Rebuild WebApp (Custom)

This is a clean reconstruction branch/folder inspired by Open WebUI interaction patterns while keeping your local/offline SCAL workflow.

## What is included

- Standalone local inference API integration (`scal_inference_api`)
- Streaming chat (`/api/chat/stream`) with token-by-token UI updates
- No forced document selection for chat
- Session-based chat history
- Auto-built TF-IDF index for extracted JSON/MD
- Assistant label uses active model name

## Run

Use one command to launch both services:

`run_scal_rebuild_stack_conda.bat`

The rebuilt UI runs at:

`http://127.0.0.1:8092`

## Notes

- Data root default is set in `scal_rebuild_webapp/main.py` and can be overridden via `SCAL_DATA_ROOT`.
- Inference URL defaults to `http://127.0.0.1:8010` and can be overridden via `SCAL_INFERENCE_API_URL`.
