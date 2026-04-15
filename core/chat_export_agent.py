from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.export_service import ChatRecord, ExportService
from core.loaders import DocumentPackage
from core.retriever import RetrievalEngine, RetrievedChunk


@dataclass(slots=True)
class ChatExportResult:
    ok: bool
    message: str
    file_path: str = ""
    export_format: str = ""
    matched_chunks: int = 0


class ChatExportAgent:
    def __init__(self, retrieval_engine: RetrievalEngine, export_service: ExportService) -> None:
        self.retrieval_engine = retrieval_engine
        self.export_service = export_service

    def run_export(
        self,
        question: str,
        export_format: str,
        package: DocumentPackage | None,
        package_id: str | None,
    ) -> ChatExportResult:
        cleaned = question.strip()
        chunks = self._retrieve_for_export(cleaned, package, package_id)
        if not chunks:
            return ChatExportResult(
                ok=False,
                message=(
                    "I could not find any relevant extracted content for this export request. "
                    "Try a more specific metric or load/select another report package."
                ),
                export_format=export_format,
                matched_chunks=0,
            )

        destination = self._build_output_path(export_format)
        records = self._chunks_to_records(question, chunks)

        if export_format == "excel":
            ok, message = self._export_excel(chunks, destination)
        elif export_format == "csv":
            ok, message = self._export_csv(chunks, destination)
        else:
            ok, message = self.export_service.export_chat_records(records, destination)

        if not ok:
            return ChatExportResult(
                ok=False,
                message=message,
                file_path=str(destination),
                export_format=export_format,
                matched_chunks=len(chunks),
            )

        return ChatExportResult(
            ok=True,
            message=message,
            file_path=str(destination),
            export_format=export_format,
            matched_chunks=len(chunks),
        )

    def _retrieve_for_export(
        self,
        question: str,
        package: DocumentPackage | None,
        package_id: str | None,
    ) -> list[RetrievedChunk]:
        if package is not None:
            direct = self.retrieval_engine.retrieve_direct(
                package=package,
                question=question,
                top_k=16,
                min_score=0.35,
                allow_fallback=True,
            )
            if direct:
                return direct

        rag = self.retrieval_engine.retrieve_rag(
            question=question,
            top_k=20,
            package_id=package_id,
            min_score=0.35,
            allow_fallback=True,
        )
        return rag

    def _chunks_to_records(self, question: str, chunks: list[RetrievedChunk]) -> list[ChatRecord]:
        records: list[ChatRecord] = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for idx, chunk in enumerate(chunks[:20], start=1):
            records.append(
                ChatRecord(
                    timestamp=now,
                    mode="export",
                    runtime="retrieval",
                    model="",
                    question=question,
                    answer=f"[{idx}] {chunk.content}",
                    citations=f"{chunk.source_file}:{chunk.source_type}:{chunk.score:.2f}",
                )
            )
        return records

    def _export_excel(self, chunks: list[RetrievedChunk], destination: Path) -> tuple[bool, str]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, chunk in enumerate(chunks, start=1):
            rows.append(
                {
                    "rank": idx,
                    "source_file": chunk.source_file,
                    "source_type": chunk.source_type,
                    "score": float(chunk.score),
                    "section": chunk.section,
                    "page": chunk.page,
                    "content": chunk.content,
                }
            )

        frame = pd.DataFrame(rows)
        summary = pd.DataFrame(
            [
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "table_count": len(rows),
                }
            ]
        )

        with pd.ExcelWriter(destination, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Summary", index=False)
            frame.to_excel(writer, sheet_name="ExtractedTables", index=False)

            for sheet in writer.sheets.values():
                for col in sheet.columns:
                    max_len = 0
                    col_letter = col[0].column_letter
                    for cell in col[:200]:
                        val = "" if cell.value is None else str(cell.value)
                        if len(val) > max_len:
                            max_len = len(val)
                    sheet.column_dimensions[col_letter].width = min(70, max(12, max_len + 2))

        return True, f"Exported Excel: {destination}"

    def _export_csv(self, chunks: list[RetrievedChunk], destination: Path) -> tuple[bool, str]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, chunk in enumerate(chunks, start=1):
            rows.append(
                {
                    "rank": idx,
                    "source_file": chunk.source_file,
                    "source_type": chunk.source_type,
                    "score": float(chunk.score),
                    "section": chunk.section,
                    "page": chunk.page,
                    "content": chunk.content,
                }
            )

        pd.DataFrame(rows).to_csv(destination, index=False)
        return True, f"Exported CSV: {destination}"

    def _build_output_path(self, export_format: str) -> Path:
        root = Path("data/exports").resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = {
            "excel": "xlsx",
            "word": "docx",
            "csv": "csv",
        }.get(export_format, "xlsx")
        return root / f"chat_export_{stamp}.{ext}"
