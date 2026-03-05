"""olmOCR Post-processor utilities

This module provides a small, dependency-light post-processing pipeline:
- Load an existing extraction (text or JSON)
- Use an LLM (local or API) to structure data into records
- Export structured data to Excel/CSV/JSON

The goal is to separate post-processing from the GUI/extraction pipeline so
you can reuse it in automation or a separate UI/workflow.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd


STRUCTURE_DEFAULT_SYSTEM = (
    "You are a precise data extraction assistant. Return only what is explicitly "
    "present in the text as a JSON array. Never fabricate or infer values."
)


def _lazy_import_local_llm():
    try:
        # Imported lazily to avoid import cycle during GUI load
        from olmocr_agentic_gui import IntelligentAssistant  # type: ignore
        return IntelligentAssistant()
    except Exception:
        return None


def _lazy_import_api_llm(provider: str, api_key: Optional[str], model: Optional[str] = None):
    try:
        from olmocr_agentic_gui import APILLM  # type: ignore
        return APILLM(provider=provider, api_key=api_key, model=model)
    except Exception:
        return None


def _build_postprocess_prompt(raw_text: str, columns: Optional[List[str]] = None) -> str:
    if columns:
        cols_str = ", ".join(columns)
        schema_instruction = (
            f"Return a JSON array of objects using EXACTLY these column names: [{cols_str}]\n"
            f"- Do not add extra columns beyond those listed\n"
            f"- Use null for any column where the value is not found in the text"
        )
    else:
        schema_instruction = (
            "Infer an appropriate schema from the content.\n"
            "- Use descriptive snake_case column names\n"
            "- Group related values into the same record where they clearly belong together\n"
            "- Return a JSON array of objects — each object is one record/row"
        )

    return f"""You are a data extraction specialist working on multi-page OCR output.\n\nThe text below is compiled from all pages of a scanned document.\nYour task: extract ALL structured records (rows of data) from this text.\n\n{schema_instruction}\n\nGeneral rules:\n- Only extract values that explicitly appear in the text — do NOT infer or hallucinate\n- Numbers must be numeric type, not strings\n- If multiple records/rows exist, include ALL of them as separate objects in the array\n- Ignore page headers, footers, and VLM metadata lines (primary_language, is_rotation_valid, etc.)\n- Return ONLY a valid JSON array — no explanation, no markdown code fences, no preamble\n\nCOMPILED DOCUMENT TEXT:\n{raw_text}""" 


def _parse_json_text(text: str, columns: Optional[List[str]] = None) -> List[dict]:
    cleaned = text.strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = [data]
        return data
    except Exception:
        return []


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        t = "\n".join(inner).strip()
    return t


def _ensure_list_of_records(data) -> List[dict]:
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    return []


def _convert_to_dataframe(records: List[dict], template_columns: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if template_columns:
        ordered = [c for c in template_columns if c in df.columns]
        extras = [c for c in df.columns if c not in template_columns]
        df = df[ordered + extras]
    return df


def _export_excel(df: pd.DataFrame, metadata: dict, filename: str) -> None:
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Data", index=False)
        meta_df = pd.DataFrame([metadata])
        meta_df.to_excel(writer, sheet_name="_metadata", index=False)


def process_extraction_text(raw_text: str, columns: Optional[List[str]] = None,
                            llm_provider: str = "local", api_key: Optional[str] = None,
                            llm_model: Optional[str] = None) -> Tuple[List[dict], str]:
    """Process raw OCR text to structured records using an LLM.

    Returns (records, structured_json_str).
    """

    prompt = _build_postprocess_prompt(raw_text, columns)
    # Local LLM
    llm = None
    if llm_provider == "local":
        llm = _lazy_import_local_llm()
        if llm is None:
            raise RuntimeError("Local LLM not available. Ensure olmocr GUI dependencies are installed.")
        try:
            llm.load_model()
        except Exception:
            pass
    else:
        llm = _lazy_import_api_llm(llm_provider, api_key, llm_model)
        if llm is None:
            raise RuntimeError("API LLM not available. Install required libraries (olmocr GUI API).")
        llm.load_model()

    system_ctx = STRUCTURE_DEFAULT_SYSTEM
    try:
        structured = llm.chat(prompt, system_context=system_ctx)
    except Exception:
        # Fallback: send a plain prompt if chat interface changed
        structured = llm.chat(raw_text, system_context=system_ctx)

    cleaned = _strip_fences(str(structured))
    records = _ensure_list_of_records(_parse_json_text(cleaned, columns))
    structured_json = json.dumps(records, indent=2)
    return records, structured_json


def process_extraction_file(file_path: str, columns: Optional[List[str]] = None,
                            llm_provider: str = "local", api_key: Optional[str] = None,
                            llm_model: Optional[str] = None) -> Tuple[List[dict], str, str]:
    """Load an extraction file and post-process it into structured records.

    Returns (records, structured_json, raw_text_combined).
    Raw text is a concatenation of all page extractions if available.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    raw_text = ""
    # Try JSON first
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Heuristic: collect all raw_response fields
        if isinstance(data, list):
            parts = []
            for item in data:
                if isinstance(item, dict) and "raw_response" in item:
                    parts.append(str(item.get("raw_response")))
            raw_text = "\n\n".join(parts)
        elif isinstance(data, dict):
            parts = []
            for k in ("raw_extraction", "raw_extractions", "extracted_text", "results", "pages"):
                if k in data:
                    v = data[k]
                    if isinstance(v, str):
                        parts.append(v)
                    elif isinstance(v, list):
                        parts.extend([str(x) for x in v if isinstance(x, (str, int, float))])
            raw_text = "\n\n".join(parts)
        if not raw_text:
            raw_text = json.dumps(data, indent=2)
        # Fall back to string content of the file if nothing else
    except Exception:
        # Treat as plain text
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

    records, structured_json = process_extraction_text(raw_text, columns, llm_provider, api_key, llm_model)
    return records, structured_json, raw_text


def export_records_to_excel(records: List[dict], filename: str, template_columns: Optional[List[str]] = None) -> None:
    df = _convert_to_dataframe(records, template_columns)
    metadata = {
        "export_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "records_extracted": len(records),
        "template_columns": template_columns or [],
    }
    _export_excel(df, metadata, filename)
