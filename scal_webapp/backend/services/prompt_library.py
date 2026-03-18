from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path("scal_webapp/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_FILE = DATA_DIR / "saved_chat_prompts.json"


DEFAULT_PROMPTS = [
    {
        "name": "Extract certain columns only",
        "text": (
            "You are a document analysis assistant.\n\n"
            "I will provide extracted JSON/table evidence from a report.\n"
            "Your task is to return ONLY the following columns if present:\n"
            "- Sample ID\n- Porosity\n- Depth\n\n"
            "Important Instructions:\n"
            "- Ignore all other columns not listed above.\n"
            "- Preserve original row order from evidence.\n"
            "- If a specified column is missing in a row, return NULL.\n"
            "- Output strictly valid JSON.\n"
            "- No explanation, no summary, no extra text."
        ),
    },
    {
        "name": "Extract table based on keyword",
        "text": (
            "I need you to extract a specific table from retrieved evidence.\n"
            "The report may contain multiple tables, but I'm interested in the one containing keyword '[keyword]'.\n\n"
            "Steps:\n"
            "1. Scan retrieved table evidence\n"
            "2. Identify table containing '[keyword]'\n"
            "3. Extract complete table with all headers, rows, and columns\n"
            "4. If table present, return ALL rows in exact order with exact column names as JSON array\n"
            "5. If no table present: {\"no_table\": true}"
        ),
    },
    {
        "name": "Graph extraction",
        "text": (
            "Perform structured visual data extraction from retrieved graph/table evidence.\n\n"
            "Steps:\n"
            "1. Identify chart type\n"
            "2. Identify axis scale intervals\n"
            "3. Determine legend-to-series mapping\n"
            "4. Extract exact values if labeled\n"
            "5. If not labeled, calculate approximate values using axis scaling\n\n"
            "Return:\n"
            "- Clean structured JSON table\n"
            "- Numeric values only (no unit text)\n"
            "- No commentary"
        ),
    },
]


def _ensure_file():
    if PROMPT_FILE.exists():
        return
    with open(PROMPT_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_PROMPTS, f, indent=2, ensure_ascii=False)


def list_prompts() -> list[dict]:
    _ensure_file()
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return DEFAULT_PROMPTS
    return data


def save_prompt(name: str, text: str) -> list[dict]:
    prompts = list_prompts()
    # replace if same name
    replaced = False
    for p in prompts:
        if p.get("name") == name:
            p["text"] = text
            replaced = True
            break
    if not replaced:
        prompts.append({"name": name, "text": text})
    with open(PROMPT_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    return prompts
