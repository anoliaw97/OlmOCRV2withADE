# OLM OCR Web GUI

Flask-based web version of the olmocr_agentic_gui.py

## Requirements

```bash
pip install flask pandas openpyxl
```

## Run

```bash
cd webapp
python app.py
```

Then open http://localhost:5000 in your browser.

## Features

- PDF Upload
- Run Extraction (demo loads Angsi data)
- Prompt Optimizer UI
- Post-Process with optional template
- Export to Excel, CSV, JSON
- Screenshots gallery
- Page-by-page navigation with timing info

## Demo Data

Click "Load Angsi Demo" to load the Angsi 1 Core analysis results.
