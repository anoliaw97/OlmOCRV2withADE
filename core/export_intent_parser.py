from __future__ import annotations

from dataclasses import dataclass


EXCEL_HINTS = {"excel", "xlsx", "spreadsheet"}
WORD_HINTS = {"word", "docx", "document"}
CSV_HINTS = {"csv"}

EXPORT_ACTION_HINTS = {
    "export",
    "save",
    "save as",
    "compile",
    "generate",
    "create",
    "output",
}


@dataclass(slots=True)
class ExportIntent:
    is_export: bool
    export_format: str
    confidence: float
    wants_tables: bool
    wants_summary: bool
    reason: str


def parse_export_intent(question: str) -> ExportIntent:
    text = " ".join((question or "").lower().split())
    if not text:
        return ExportIntent(
            is_export=False,
            export_format="",
            confidence=0.0,
            wants_tables=False,
            wants_summary=False,
            reason="empty-question",
        )

    has_export_verb = any(token in text for token in EXPORT_ACTION_HINTS)
    format_name = _resolve_format(text)
    wants_tables = any(token in text for token in ("table", "tables", "spreadsheet"))
    wants_summary = any(token in text for token in ("summary", "summarize", "explain", "findings"))

    if not has_export_verb and not format_name:
        return ExportIntent(
            is_export=False,
            export_format="",
            confidence=0.0,
            wants_tables=wants_tables,
            wants_summary=wants_summary,
            reason="no-export-signals",
        )

    if not format_name:
        if wants_tables:
            format_name = "excel"
        elif wants_summary:
            format_name = "word"
        else:
            format_name = "excel"

    confidence = 0.86 if has_export_verb else 0.72
    if wants_tables or wants_summary:
        confidence = min(0.95, confidence + 0.05)

    return ExportIntent(
        is_export=True,
        export_format=format_name,
        confidence=confidence,
        wants_tables=wants_tables,
        wants_summary=wants_summary,
        reason="export-intent-detected",
    )


def _resolve_format(text: str) -> str:
    if any(token in text for token in EXCEL_HINTS):
        return "excel"
    if any(token in text for token in WORD_HINTS):
        return "word"
    if any(token in text for token in CSV_HINTS):
        return "csv"
    return ""
