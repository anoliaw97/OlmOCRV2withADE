# Python Workflow WebApp

Webapp-only local workflow for querying extracted document outputs with a FastAPI backend.

This project is designed for Windows + Anaconda Prompt.

## Core rules

- Source of truth for retrieval/chat: **JSON / Markdown / TXT extracted files only**
- PDF and image files: **preview-only**
- Extraction scripts remain separate from this app

## Architecture

- FastAPI backend with modular routers:
  - loaders
  - preview
  - retrieval/index
  - chat
  - export
  - system directory browse
- Browser frontend served by FastAPI (`webapp/index.html`)
- SQLite + FTS5 optional local retrieval index (`data/index/rag_index.sqlite`)

## Enhanced web UX features

- Directory browser defaults to `C:\Users\admin\Downloads\Fine Tunining Datasets\train`
- Page-level files (for example `report_page1.pdf`, `report_page2.pdf`, ...) are grouped into one report package
- Model discovery for:
  - Ollama local tags/models
  - llama.cpp GGUF files discovered from configured scan path (default `D:\models`)
- PDF preview rendering uses Poppler (`pdftoppm`) for page images inside the web viewer
- Chat response metrics include context limit/source, retrieval and generation timings, and total duration
- Assistant display name follows selected model name
- Persistent chat sessions (`data/chat_sessions.json`)
- Advanced panel includes live Debug / Status / Error / Reasoning logs

## Project structure

```text
G:\Python Based Workflow\
|- app.py
|- run_backend.py
|- requirements.txt
|- README.md
|- backend/
|  |- app.py
|  |- dependencies.py
|  |- runtime.py
|  |- schemas.py
|  |- utils.py
|  `- api/routers/
|- core/
|  |- loaders.py
|  |- preview_service.py
|  |- markdown_service.py
|  |- table_renderer.py
|  |- json_chunker.py
|  |- retriever.py
|  |- rag_index.py
|  |- chat_agent.py
|  |- export_service.py
|  |- pdf_preview.py
|  `- runtime_logs.py
|- webapp/
|  |- index.html
|  `- assets/
|     |- styles.css
|     `- app.js
`- examples/
```

## Setup (Anaconda Prompt)

### 1) Clone branch

```bat
cd /d "C:\Users\admin\Downloads"
git clone -b feature/python-desktop-workflow-v1 https://github.com/anoliaw97/OlmOCRV2withADE.git "Python Based Workflow"
cd /d "C:\Users\admin\Downloads\Python Based Workflow"
```

### 2) Create environment

```bat
conda create -n python_workflow python=3.11 -y
conda activate python_workflow
```

### 3) Install dependencies

```bat
pip install -r requirements.txt
```

### 3.1) Auto-bootstrap local tools (recommended)

Run this once from Anaconda Prompt (inside the same env):

```bat
python bootstrap_tools.py
```

What it does:

- Downloads and installs Poppler into `tools\poppler` (if missing)
- Sets `POPPLER_PDFTOPPM` for current shell
- Persists `POPPLER_PDFTOPPM` via `setx` for future shells
- Checks whether `llama-cli` is discoverable on PATH

If you do not want persistent env var updates:

```bat
python bootstrap_tools.py --no-setx
```

### 3.2) Install Poppler manually (alternative)

- Install Poppler for Windows and ensure `pdftoppm.exe` is available in `PATH`,
  or set `POPPLER_PDFTOPPM` to full executable path.
- Common path example:
  - `C:\Program Files\poppler\Library\bin\pdftoppm.exe`

### 3.3) llama.cpp setup (if using llama.cpp backend)

- Set `llama-cli path` in UI to full executable path, for example:
  - `C:\llama.cpp\build\bin\Release\llama-cli.exe`
- If not set, chat will refuse to run with llama.cpp backend to avoid fallback confusion.

Optional transformers runtime:

```bat
pip install transformers torch accelerate safetensors sentencepiece
```

### 4) Run webapp

```bat
python run_backend.py
```

Open in browser:

- `http://127.0.0.1:8000`

`app.py` now starts the same backend entry flow for compatibility.

## First workflow

1. Open `http://127.0.0.1:8000`
2. Step 1: `Load Folder` (or `Load Database`) for extracted outputs
3. Step 2: `Build / Rebuild RAG`
4. Step 3: choose backend/model and click `Load Model`
5. Step 4: ask in Chat
6. Step 5: review generated/arranged tables from answers
7. Step 6: export chat to Excel or Word
8. Use `Preview` tab to inspect:
   - Poppler-rendered PDF page preview
   - Raw Markdown/JSON/TXT extracted content
   - Rendered extracted tables
9. Use right-side advanced panel to monitor:
   - Status logs
   - Debug logs
   - Error logs
   - Reasoning detail logs

## Crash stabilization included

- Bounded JSON flattening for chunking to avoid runaway recursion/memory on deep extracted payloads
- Bounded JSON table-walk extraction to avoid oversized nested table parsing
- Fail-safe preview table extraction when malformed table structures are encountered
- SQLite index access made thread-safe for backend API requests

## Notes

- Supported companion extensions: `.pdf`, `.json`, `.md`, `.markdown`, `.txt`
- Folder loader scans selected folder root (non-recursive)
- RAG index is lexical (FTS), not embedding-vector semantic search
- LLM runtime scope is local-only in this workflow: Ollama or llama.cpp

## Optional Streamlit Variant (LangChain + HF Embeddings + Chroma + PyPDFLoader)

An optional prototype app is included to mirror the architecture style you referenced:

- LangChain RAG pipeline and optional agent mode
- HuggingFace embeddings
- ChromaDB vector store
- PyPDFLoader document processing

Files:

- `streamlit_option_app.py`
- `requirements-streamlit.txt`

Install in current environment:

```bat
pip install -r requirements-streamlit.txt
```

Run:

```bat
streamlit run streamlit_option_app.py
```

Notes:

- This is additive and does not replace FastAPI webapp flow.
- Streamlit variant expects PDF-first ingestion (not extracted JSON/MD/TXT pairing).
- For llama.cpp backend in Streamlit, install `llama-cpp-python` and use a local GGUF model path.

## ML predictive analytics integration (web app)

The main web app now includes an ML Analytics tab for automated predictive workflow from extracted sources:

1. Build structured dataset from loaded extracted JSON/MD/TXT report packages
2. Train regression model for selected target (for example relative permeability feature)
3. Run prediction from input feature JSON
4. View dashboard summary + scatter analytics
5. Run YAML-driven ML pipeline definition (`/api/ml/pipeline/run`)

Available API endpoints:

- `POST /api/ml/dataset/build`
- `POST /api/ml/train`
- `POST /api/ml/predict`
- `GET /api/ml/dashboard`
- `POST /api/ml/pipeline/run`

## Chat-driven export actions (Step 6 upgraded)

Export is now a first-class chat-agent action. You can request exports in natural language from chat, for example:

- "Export all permeability tables to Excel"
- "Compile all SCAL tables and save as xlsx"
- "Summarize this report and export to Word"

Behavior:

1. Chat intent parser detects export action + target format
2. Retrieval gathers relevant extracted JSON/MD/TXT chunks
3. Export agent compiles content and generates file
4. Chat returns success/failure with file path and action metadata

Export files are generated under:

- `data/exports/`

You can also choose custom save destination:

- In chat requests, destination uses the Step 6 output folder input when provided
- If destination is a folder, file name is auto-generated
- If destination is a full file path, that exact path is used

Chat export supports:

- Excel (`.xlsx`)
- CSV (`.csv`)
- Word (`.docx`)

## Poppler configuration in app

If PDF preview reports missing `pdftoppm`, use the ML Analytics tab controls:

1. Set full Poppler path (for example `C:\tools\poppler\Library\bin\pdftoppm.exe`)
2. Click `Set Poppler`
3. Click `Check` to verify resolved executable

You can still use bootstrap command:

```bat
python bootstrap_tools.py
```

Added dependencies:

- `scikit-learn`
- `numpy`
- `pyyaml`
