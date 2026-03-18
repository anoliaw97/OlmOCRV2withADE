# SCAL Extraction Web App (Offline, Open Source)

This is a deployable web app starter with two connected modules:

1. **Extraction Module** (new PDF full extraction + JSON import)
2. **Chat/RAG Module** (answers from extracted JSON/chunks only via local LLM)

## Hard Rule Enforced

The chat service **never reads raw PDFs**. It only queries indexed chunks produced from extracted table JSON.

## Updated Workflow

1. Upload new PDF -> default full extraction (page-level JSON records)
2. Or import existing extraction JSON (single-page or multi-page)
3. Build/refresh RAG index from extracted JSON
4. Ask questions in chat; optional use-case prompt uses second local LLM pass
5. Answer returns source traceability (`file_name`, `page_number`, `table_id`)

## Features

- Layman mode and Operator mode in one UI
- Targeted SCAL extraction types:
  - capillary pressure
  - relative permeability
  - porosity/permeability
- 1 table = 1 JSON object with metadata fields
- Existing JSON import for previously extracted reports
- Default full extraction for new PDFs, use-case filtering at chat-time over indexed JSON
- Saved prompt library for chat-side JSON querying (load/save prompt templates)
- Post-processing:
  - normalize column names
  - merge duplicate columns
  - missing value handling (`null`)
  - ML-ready dataframe output
  - RAG-ready chunks
- Hybrid retrieval (keyword + local vector TF-IDF)
- Source traceability in answers (`file_name`, `page_number`, `table_id`)
- Local LLM-driven RAG answer generation (offline)
- Timestamped processing/indexing logs with clear history button
- Export to JSON, Excel, Word

## Architecture

- **Backend:** FastAPI + SQLAlchemy + SQLite (offline)
- **RAG Index:** local TF-IDF vector index (`scikit-learn`) + metadata filter
- **UI:** server-rendered HTML + vanilla JS (internal-use friendly)
- **Storage:** local filesystem (`scal_webapp/data`)

This starter is fully open source and offline. No external API is required.

## Folder Structure

```txt
scal_webapp/
  backend/
    main.py
    database.py
    models.py
    schemas.py
    routers/
      ui.py
      extraction.py
      chat.py
    services/
      extractor.py
      postprocess.py
      indexer.py
      rag.py
      exporter.py
      logger.py
    templates/index.html
    static/app.js
    static/styles.css
  data/
  requirements.txt
  Dockerfile
  docker-compose.yml
```

## Data Model (current implementation)

- `reports`
- `extracted_tables`
- `rag_chunks`
- `processing_logs`

## Vector DB Design (current + future)

Current offline implementation:
- local TF-IDF matrix with metadata payloads (stored via `joblib`)

Future scalable swap-in:
- Qdrant collection with payload filters (`file_name`, `page_number`, `table_id`, `extraction_type`, `sample_id`, `report_name`)

## Run Locally

```bash
cd scal_webapp
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn scal_webapp.backend.main:app --host 0.0.0.0 --port 8080
```

Open: `http://localhost:8080`

## One-command launcher

From repo root:

```bash
python -m scal_webapp
```

Windows batch launcher:

```bash
run_scal_webapp.bat
```

## Package and install anywhere

Build wheel package on a source machine:

```bash
build_scal_webapp_package.bat
```

This creates package files in `scal_webapp/dist/`.

On any target machine (offline/internal copy supported):

```bash
install_scal_webapp_package.bat
```

Or pass an explicit wheel path:

```bash
install_scal_webapp_package.bat C:\path\to\scal_webapp_offline-0.1.0-py3-none-any.whl
```

Then run from any directory:

```bash
scal-webapp
```

## Full one-shot rebuild (Conda `olmocr`)

From repository root (Windows):

```bash
rebuild_olmocr_all_in_one.bat
```

This script will:
- recreate `olmocr` conda environment from scratch
- install all required packages
- install Poppler
- install CUDA PyTorch build
- install local `scal_webapp` package
- pre-download required Hugging Face models

Skip model downloads if needed:

```bash
rebuild_olmocr_all_in_one.bat --skip-model-download
```

## Run with Docker

```bash
cd scal_webapp
docker compose up --build
```

## Operator Controls

- page range
- extraction types
- prompt profile
- model field
- post-processing toggles
- index rebuild toggle
- debug logs panel

## Experiment Evaluation Tiers

- **Easy:** >=300 DPI, clear borders, simple tables
- **Medium:** 200-299 DPI, moderate noise, multi-row headers
- **Difficult:** <200 DPI, poor scans, broken borders, irregular layouts

Recommended metrics:
- table detection recall
- row/column parse accuracy
- numeric fidelity
- RAG citation correctness

## Implementation Roadmap

1. Replace heuristic extractor with your existing olmOCR page output parser
2. Add robust table parser (Camelot/Tabula/vision parser)
3. Introduce async jobs (Celery + Redis)
4. Move SQLite to PostgreSQL for multi-user deployment
5. Switch TF-IDF index to Qdrant for scale
6. Add RBAC and SSO for internal users
