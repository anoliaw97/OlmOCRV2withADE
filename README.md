# Python Workflow Query Desktop

Local Python workflow app with a FastAPI backend plus a PySide6 desktop UI for loading extracted document outputs and querying them with a grounded chat workflow.

This project is designed for Windows + Anaconda Prompt and keeps extraction separate from query/chat.

## LLM runtime support (now included)

The chat panel supports runtime selection:

- **Ollama** (local running service, model name like `llama3.1:8b`)
- **llama.cpp (GGUF)** (local `.gguf` path + `llama-cli` executable)
- **Transformers (tensor/safetensors)** (local model folder or model id)
- **Heuristic (no model)** fallback

Important:

- Answers remain grounded in extracted JSON/MD/TXT context.
- PDF and image assets are still preview-only.

## What this app is for

- Load extracted document packages (JSON / MD / TXT + optional PDF companion).
- Preview package content in practical tabs.
- Chat over extracted outputs only (no PDF/image content in answer generation).
- Optionally build/update a lightweight local retrieval index.
- Export chat outputs to CSV / Excel / Word.

## Source-of-truth rules in this app

- Chat answers use **JSON / Markdown / TXT** only.
- PDF is **preview only**.
- Images are not used as chat knowledge source.

## Minimal practical architecture adjustment

To keep the first version simple and local, optional RAG indexing is implemented with:

- SQLite + FTS5 (Python-native, no external vector DB service required)

This can later be replaced by FAISS/Chroma without rewriting the UI.

## Project structure

```text
G:\Python Based Workflow\
|- app.py
|- run_backend.py
|- requirements.txt
|- README.md
|- .gitignore
|- backend/
|  |- __init__.py
|  |- app.py
|  |- dependencies.py
|  |- runtime.py
|  |- schemas.py
|  |- utils.py
|  `- api/
|     |- __init__.py
|     `- routers/
|        |- chat.py
|        |- export.py
|        |- loaders.py
|        |- retrieval.py
|        `- __init__.py
|- config/
|- data/
|- examples/
|- core/
|  |- loaders.py
|  |- preview_service.py
|  |- markdown_service.py
|  |- table_renderer.py
|  |- json_chunker.py
|  |- retriever.py
|  |- chat_agent.py
|  |- export_service.py
|  `- rag_index.py
|- ui/
|  |- main_window.py
|  `- dialogs.py
`- widgets/
   |- chat_widget.py
   |- pdf_viewer.py
   |- markdown_viewer.py
   |- json_viewer.py
   `- table_viewer.py
```

## Setup (Anaconda Prompt)

### 0) Clone repository

```bat
cd /d "C:\Users\admin\Downloads"
git clone -b feature/python-desktop-workflow-v1 https://github.com/anoliaw97/OlmOCRV2withADE.git "Python Based Chat Agent"
cd /d "C:\Users\admin\Downloads\Python Based Chat Agent"
```

### 1) Create and activate env

```bat
conda create -n python_workflow python=3.11 -y
conda activate python_workflow
```

### 2) Go to project folder

```bat
cd /d "C:\Users\admin\Downloads\Python Based Chat Agent"
```

### 3) Install dependencies

```bat
pip install -r requirements.txt
```

### 3b) Optional dependencies for tensor/safetensors models (Transformers backend)

Install only if you want transformers local runtime:

```bat
pip install transformers torch accelerate safetensors sentencepiece
```

### 4) Start FastAPI backend

```bat
python run_backend.py
```

Backend API is served at `http://127.0.0.1:8000`.

### 5) Run desktop UI (separate Anaconda Prompt)

```bat
python app.py
```

The desktop UI calls backend APIs for loaders/chat/retrieval/export operations.

## First workflow (end-to-end)

1. Click **Load Folder** and choose a directory containing extracted outputs.
   - Quick demo folder in this repo: `C:\Users\admin\Downloads\Python Based Chat Agent\examples`
2. Select a detected package from the left list.
3. Use tabs on the right:
   - **PDF Preview**: opens PDF in system viewer
   - **Markdown Preview**
   - **Rendered Tables**
   - **Raw JSON**
   - **Chat**
4. In **Chat** tab:
   - choose mode (**Direct** or **Optional indexed RAG**)
   - choose **LLM Backend** (`Auto`, `Ollama`, `llama.cpp`, `Transformers`, or `Heuristic`)
   - set model information:
     - Ollama: model name (example: `llama3.1:8b`)
     - llama.cpp: local `.gguf` model path and `llama-cli` path
     - Transformers: local model folder path containing tensor/safetensors model files
   - ask question (e.g., porosity/permeability/SCAL)
5. Optional: click **Build/Update Optional RAG Index** for multi-package retrieval.
6. Export chat using **Export chat...** (CSV/XLSX/DOCX).

## Supported package detection

The loader groups related files by normalized stem in one folder.

Example companions:

- `report.pdf`
- `report.json`
- `report.md`
- `report.txt`

Supported extensions:

- `.pdf`, `.json`, `.md`, `.markdown`, `.txt`

## Current implementation status

### Working now

- Modular FastAPI backend (`backend/app.py`) with API endpoints for:
  - loaders
  - preview
  - retrieval/index
  - chat ask
  - export
- Runnable PySide6 desktop app shell.
- Folder/file package loading and auto grouping.
- Preview tabs wired and functional.
- JSON viewer + Markdown HTML preview.
- Table detection/rendering from:
  - HTML tables embedded in markdown/json strings
  - markdown table blocks
  - JSON list-of-dict table-shaped data
- Direct selected-document chat over extracted JSON/MD/TXT.
- LLM-backed grounded answering with runtime selection:
  - Ollama
  - llama.cpp for GGUF
  - Transformers for tensor/safetensors
  - heuristic fallback
- Optional local index build/update and retrieval mode (SQLite FTS).
- Chat export to CSV, Excel, Word.

### Explicit placeholders / next-step enhancements

- Embedded PDF rendering is not implemented yet; app uses external PDF open strategy.
- RAG index is lightweight lexical retrieval, not embedding-based semantic retrieval.
- Source citations are basic and can be expanded with richer provenance UI.

## Crash stabilization notes

Recent runtime hardening added safeguards for large extracted payloads:

- Bounded JSON flattening during chunking to avoid runaway memory/recursion on deeply nested extraction files.
- Bounded table extraction when walking JSON structures and list-of-dict tables.
- Preview table extraction now fails safe (empty table list) instead of raising to the UI when malformed structures are encountered.

## Notes about extraction scripts

Extraction remains separate by design.

Reference scripts (unchanged):

- `C:\Users\Mining\Downloads\olmocr-main\olmocr_trainer.py`
- `C:\Users\Mining\Downloads\olmocr-main\olmocr_agentic_gui.py`

This app consumes extraction outputs from those workflows.

## Troubleshooting

### Clone into a specific remote PC folder

If you need an exact target folder name:

```bat
cd /d "C:\Users\admin\Downloads"
git clone https://github.com/anoliaw97/OlmOCRV2withADE.git "Python Based Chat Agent"
cd /d "C:\Users\admin\Downloads\Python Based Chat Agent"
git checkout feature/python-desktop-workflow-v1
```

### App does not launch

- Confirm env activated: `conda activate python_workflow`
- Confirm dependencies: `pip install -r requirements.txt`

### No packages detected in folder

- Check files use supported extensions.
- Ensure files are in the selected folder root (current loader does not recurse yet).

### RAG search returns weak results

- Build/update index after loading packages.
- Use more specific query terms.

### Ollama backend fails

- Ensure Ollama is installed and running: `ollama serve`
- Pull and test a model manually:
  - `ollama run llama3.1:8b`
- Confirm URL in app is reachable (default: `http://127.0.0.1:11434/api/generate`)

### llama.cpp GGUF backend fails

- Ensure `llama-cli` is built and available.
- Set full `llama-cli` path in app if not on PATH.
- Ensure selected model file is `.gguf` and path exists.

### Transformers backend fails

- Install optional dependencies:
  - `pip install transformers torch accelerate safetensors sentencepiece`
- Ensure model folder contains required tokenizer/model files.
- Large tensor models may require significant RAM/VRAM.

### Word export fails

- Ensure `python-docx` is installed from requirements.
