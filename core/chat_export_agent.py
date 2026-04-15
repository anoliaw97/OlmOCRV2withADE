from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.export_service import ChatRecord, ExportService
from core.loaders import DocumentPackage
from core.retriever import RetrievalEngine, RetrievedChunk
from core.table_renderer import ExtractedTable, extract_tables_from_json_text, extract_tables_from_markdown


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
        destination: str = "",
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

        destination_path = self._build_output_path(export_format, destination)
        records = self._chunks_to_records(question, chunks)

        if export_format == "excel":
            ok, message = self._export_excel(chunks, cleaned, destination_path)
        elif export_format == "csv":
            ok, message = self._export_csv(chunks, destination_path)
        else:
            ok, message = self.export_service.export_chat_records(records, destination_path)

        if not ok:
            return ChatExportResult(
                ok=False,
                message=message,
                file_path=str(destination_path),
                export_format=export_format,
                matched_chunks=len(chunks),
            )

        return ChatExportResult(
            ok=True,
            message=message,
            file_path=str(destination_path),
            export_format=export_format,
            matched_chunks=len(chunks),
        )

    def _retrieve_for_export(
        self,
        question: str,
        package: DocumentPackage | None,
        package_id: str | None,
    ) -> list[RetrievedChunk]:
        rag = self.retrieval_engine.retrieve_rag(
            question=question,
            top_k=24,
            package_id=None,
            min_score=0.35,
            allow_fallback=True,
        )
        if rag:
            return rag

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
        return []

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

    def _export_excel(self, chunks: list[RetrievedChunk], query: str, destination: Path) -> tuple[bool, str]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tables = self._extract_structured_tables(chunks)
        source_files = sorted({str(chunk.source_file) for chunk in chunks})
        summary = pd.DataFrame([
            {
                "query": query,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "matched_chunks": len(chunks),
                "table_count": len(tables),
                "source_files": " | ".join(source_files[:50]),
            }
        ])
        sources = pd.DataFrame([
            {
                "source_file": chunk.source_file,
                "source_type": chunk.source_type,
                "score": float(chunk.score),
                "section": chunk.section,
                "page": chunk.page,
            }
            for chunk in chunks
        ])

        with pd.ExcelWriter(destination, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Metadata", index=False)
            sources.to_excel(writer, sheet_name="Sources", index=False)

            if tables:
                for idx, item in enumerate(tables, start=1):
                    sheet_name = self._safe_sheet_name(f"{idx:02d}_{item['title']}")
                    item["frame"].to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                fallback = pd.DataFrame([
                    {
                        "rank": idx,
                        "source_file": chunk.source_file,
                        "source_type": chunk.source_type,
                        "score": float(chunk.score),
                        "section": chunk.section,
                        "page": chunk.page,
                        "content": chunk.content,
                    }
                    for idx, chunk in enumerate(chunks, start=1)
                ])
                fallback.to_excel(writer, sheet_name="ExtractedContent", index=False)

            for sheet in writer.sheets.values():
                for col in sheet.columns:
                    max_len = 0
                    col_letter = col[0].column_letter
                    for cell in col[:200]:
                        val = "" if cell.value is None else str(cell.value)
                        if len(val) > max_len:
                            max_len = len(val)
                    sheet.column_dimensions[col_letter].width = min(70, max(12, max_len + 2))
                sheet.freeze_panes = "A2"

        return True, f"Exported Excel: {destination}"

    def _extract_structured_tables(self, chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
        extracted: list[dict[str, Any]] = []
        for chunk in chunks:
            tables: list[ExtractedTable] = []
            if chunk.source_type == "json":
                tables = extract_tables_from_json_text(chunk.content, chunk.source_file)
            elif chunk.source_type in {"markdown", "md", "txt", "html"}:
                tables = extract_tables_from_markdown(chunk.content, chunk.source_file)
            else:
                tables = extract_tables_from_markdown(chunk.content, chunk.source_file)

            for table in tables:
                frame = self._normalize_table_frame(table.dataframe)
                if frame is None or frame.empty:
                    continue
                extracted.append({
                    "title": table.title or chunk.source_file,
                    "frame": frame,
                })
                if len(extracted) >= 30:
                    return extracted
        return extracted

    def _normalize_table_frame(self, frame: pd.DataFrame | None) -> pd.DataFrame | None:
        if frame is None or frame.empty:
            return None

        normalized = frame.copy()
        normalized = normalized.fillna("")
        normalized.columns = [str(col).strip() or f"col_{idx + 1}" for idx, col in enumerate(normalized.columns)]
        normalized = normalized.replace(r"^\s*$", "", regex=True)

        keep_cols = [col for col in normalized.columns if (normalized[col] != "").any()]
        normalized = normalized[keep_cols] if keep_cols else normalized
        normalized = normalized.loc[(normalized != "").any(axis=1)]

        if normalized.empty:
            return None
        return normalized.reset_index(drop=True)

    def _safe_sheet_name(self, raw: str) -> str:
        cleaned = "".join(ch for ch in str(raw) if ch not in "[]:*?/\\")
        cleaned = " ".join(cleaned.split()).strip() or "Table"
        return cleaned[:31]

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

    def _build_output_path(self, export_format: str, destination: str) -> Path:
        if destination.strip():
            path = Path(destination.strip()).expanduser().resolve()
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
                return path
            ext = {
                "excel": "xlsx",
                "word": "docx",
                "csv": "csv",
            }.get(export_format, "xlsx")
            path.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return path / f"chat_export_{stamp}.{ext}"

        root = Path("data/exports").resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = {
            "excel": "xlsx",
            "word": "docx",
            "csv": "csv",
        }.get(export_format, "xlsx")
        return root / f"chat_export_{stamp}.{ext}"
