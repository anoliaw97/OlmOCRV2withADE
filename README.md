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
|  `- export_service.py
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
2. Set folder path (or file path) for extracted outputs
3. Click `Load Folder` or `Load File`
4. Select a package from the left list
5. Preview tabs:
   - Markdown
   - Tables
   - Raw JSON
   - PDF path (preview-only)
6. Ask questions in Chat (direct or rag mode)
7. Optional: build/update RAG index
8. Export chat to CSV/XLSX/DOCX via destination path

## Crash stabilization included

- Bounded JSON flattening for chunking to avoid runaway recursion/memory on deep extracted payloads
- Bounded JSON table-walk extraction to avoid oversized nested table parsing
- Fail-safe preview table extraction when malformed table structures are encountered
- SQLite index access made thread-safe for backend API requests

## Notes

- Supported companion extensions: `.pdf`, `.json`, `.md`, `.markdown`, `.txt`
- Folder loader scans selected folder root (non-recursive)
- RAG index is lexical (FTS), not embedding-vector semantic search
