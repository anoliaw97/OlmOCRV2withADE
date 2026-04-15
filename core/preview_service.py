from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.loaders import DocumentPackage
from core.markdown_service import plain_text_to_html, render_markdown_to_html
from core.table_renderer import ExtractedTable, extract_tables_from_json_text, extract_tables_from_markdown


MAX_PREVIEW_CHARS = 2_000_000


@dataclass(slots=True)
class PackagePreview:
    markdown_text: str
    markdown_html: str
    json_text: str
    text_text: str
    tables: list[ExtractedTable]
    pdf_path: Path | None


class PreviewService:
    def build_preview(self, package: DocumentPackage) -> PackagePreview:
        json_paths = package.json_paths or ([package.json_path] if package.json_path else [])
        markdown_paths = package.markdown_paths or ([package.markdown_path] if package.markdown_path else [])
        text_paths = package.text_paths or ([package.text_path] if package.text_path else [])

        json_text = _combine_text_sources(json_paths)
        markdown_text = _combine_text_sources(markdown_paths)
        text_text = _combine_text_sources(text_paths)

        if markdown_text:
            markdown_html = render_markdown_to_html(markdown_text)
        elif text_text:
            markdown_html = plain_text_to_html(text_text)
        elif json_text:
            markdown_html = plain_text_to_html(json_text)
        else:
            markdown_html = render_markdown_to_html("")

        tables: list[ExtractedTable] = []
        try:
            if markdown_text:
                source_name = package.markdown_path.name if package.markdown_path else "markdown"
                tables.extend(extract_tables_from_markdown(markdown_text, source_name))
            if json_text:
                source_name = package.json_path.name if package.json_path else "json"
                tables.extend(extract_tables_from_json_text(json_text, source_name))
        except Exception:
            tables = []

        return PackagePreview(
            markdown_text=markdown_text,
            markdown_html=markdown_html,
            json_text=json_text,
            text_text=text_text,
            tables=tables,
            pdf_path=package.pdf_path,
        )


def _safe_read_text(path: Path | None) -> str:
    if path is None:
        return ""
    if not path.exists() or not path.is_file():
        return ""

    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) <= MAX_PREVIEW_CHARS:
        return text

    clipped = text[:MAX_PREVIEW_CHARS]
    clipped += "\n\n[preview truncated: file too large for full in-memory render]"
    return clipped


def _combine_text_sources(paths: list[Path]) -> str:
    if not paths:
        return ""

    pieces: list[str] = []
    total = 0
    for idx, path in enumerate(paths, start=1):
        text = _safe_read_text(path)
        if not text:
            continue

        header = f"\n\n--- source {idx}: {path.name} ---\n"
        candidate = header + text
        if total + len(candidate) > MAX_PREVIEW_CHARS:
            remaining = MAX_PREVIEW_CHARS - total
            if remaining > 0:
                pieces.append(candidate[:remaining])
            pieces.append("\n\n[preview truncated: too many extracted files for single preview]")
            break

        pieces.append(candidate)
        total += len(candidate)

    return "".join(pieces).strip()
