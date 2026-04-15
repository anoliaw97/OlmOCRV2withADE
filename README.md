# Python Workflow Query Desktop

Local Python desktop app for loading extracted document outputs and querying them with a grounded chat workflow.

This project is designed for Windows + Anaconda Prompt and keeps extraction separate from query/chat.

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
|- requirements.txt
|- README.md
|- .gitignore
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

### 1) Create and activate env

```bat
conda create -n python_workflow python=3.11 -y
conda activate python_workflow
```

### 2) Go to project folder

```bat
cd /d "G:\Python Based Workflow"
```

### 3) Install dependencies

```bat
pip install -r requirements.txt
```

### 4) Run app

```bat
python app.py
```

## First workflow (end-to-end)

1. Click **Load Folder** and choose a directory containing extracted outputs.
   - Quick demo folder in this repo: `G:\Python Based Workflow\examples`
2. Select a detected package from the left list.
3. Use tabs on the right:
   - **PDF Preview**: opens PDF in system viewer
   - **Markdown Preview**
   - **Rendered Tables**
   - **Raw JSON**
   - **Chat**
4. In **Chat** tab:
   - choose mode (**Direct** or **Optional indexed RAG**)
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

- Runnable PySide6 desktop app shell.
- Folder/file package loading and auto grouping.
- Preview tabs wired and functional.
- JSON viewer + Markdown HTML preview.
- Table detection/rendering from:
  - HTML tables embedded in markdown/json strings
  - markdown table blocks
  - JSON list-of-dict table-shaped data
- Direct selected-document chat over extracted JSON/MD/TXT.
- Optional local index build/update and retrieval mode (SQLite FTS).
- Chat export to CSV, Excel, Word.

### Explicit placeholders / next-step enhancements

- Embedded PDF rendering is not implemented yet; app uses external PDF open strategy.
- Chat synthesis is lightweight heuristic grounding (no LLM call yet).
- RAG index is lightweight lexical retrieval, not embedding-based semantic retrieval.
- Source citations are basic and can be expanded with richer provenance UI.

## Notes about extraction scripts

Extraction remains separate by design.

Reference scripts (unchanged):

- `C:\Users\Mining\Downloads\olmocr-main\olmocr_trainer.py`
- `C:\Users\Mining\Downloads\olmocr-main\olmocr_agentic_gui.py`

This app consumes extraction outputs from those workflows.

## Troubleshooting

### App does not launch

- Confirm env activated: `conda activate python_workflow`
- Confirm dependencies: `pip install -r requirements.txt`

### No packages detected in folder

- Check files use supported extensions.
- Ensure files are in the selected folder root (current loader does not recurse yet).

### RAG search returns weak results

- Build/update index after loading packages.
- Use more specific query terms.

### Word export fails

- Ensure `python-docx` is installed from requirements.
