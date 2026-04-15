from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class ChatRecord:
    timestamp: str
    mode: str
    question: str
    answer: str
    citations: str


class ExportService:
    def export_chat_records(self, records: list[ChatRecord], destination: Path) -> tuple[bool, str]:
        if not records:
            return False, "No chat history available to export."

        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.suffix.lower() == ".csv":
            return self._to_csv(records, destination)

        if destination.suffix.lower() in {".xlsx", ".xls"}:
            return self._to_excel(records, destination)

        if destination.suffix.lower() == ".docx":
            return self._to_docx(records, destination)

        return False, f"Unsupported export format: {destination.suffix}"

    def _to_csv(self, records: list[ChatRecord], destination: Path) -> tuple[bool, str]:
        frame = pd.DataFrame([asdict(r) for r in records])
        frame.to_csv(destination, index=False)
        return True, f"Exported CSV: {destination}"

    def _to_excel(self, records: list[ChatRecord], destination: Path) -> tuple[bool, str]:
        frame = pd.DataFrame([asdict(r) for r in records])
        frame.to_excel(destination, index=False)
        return True, f"Exported Excel: {destination}"

    def _to_docx(self, records: list[ChatRecord], destination: Path) -> tuple[bool, str]:
        try:
            from docx import Document
        except ImportError:
            return False, "python-docx is not installed. Install requirements again and retry."

        document = Document()
        document.add_heading("Chat Export", level=1)

        for idx, record in enumerate(records, start=1):
            document.add_heading(f"Turn {idx}", level=2)
            document.add_paragraph(f"Timestamp: {record.timestamp}")
            document.add_paragraph(f"Mode: {record.mode}")
            document.add_paragraph(f"Question: {record.question}")
            document.add_paragraph("Answer:")
            document.add_paragraph(record.answer)
            if record.citations.strip():
                document.add_paragraph(f"Citations: {record.citations}")

        document.save(destination)
        return True, f"Exported Word document: {destination}"
