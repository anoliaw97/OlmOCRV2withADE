from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.loaders import DocumentPackage


SECTION_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(?P<title>.+)$", re.MULTILINE)
PAGE_PATTERN = re.compile(r"\bpage\s*(\d{1,4})\b", re.IGNORECASE)

MAX_JSON_FLATTEN_LINES = 120000
MAX_JSON_FLATTEN_DEPTH = 24
MAX_JSON_VALUE_CHARS = 4000


@dataclass(slots=True)
class TextChunk:
    package_id: str
    source_file: str
    source_type: str
    content: str
    chunk_index: int
    metadata: dict[str, str] = field(default_factory=dict)


def chunk_package_content(
    package: DocumentPackage,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []

    if package.json_path and package.json_path.exists():
        json_chunks = _chunk_json_file(package.package_id, package.json_path, chunk_size, overlap)
        chunks.extend(json_chunks)

    if package.markdown_path and package.markdown_path.exists():
        md_chunks = _chunk_markdown_file(package.package_id, package.markdown_path, chunk_size, overlap)
        chunks.extend(md_chunks)

    if package.text_path and package.text_path.exists():
        txt = package.text_path.read_text(encoding="utf-8", errors="ignore")
        txt_chunks = _split_text_into_chunks(
            package_id=package.package_id,
            source_file=package.text_path.name,
            source_type="txt",
            text=txt,
            chunk_size=chunk_size,
            overlap=overlap,
            base_metadata={},
        )
        chunks.extend(txt_chunks)

    return chunks


def _chunk_json_file(package_id: str, path: Path, chunk_size: int, overlap: int) -> list[TextChunk]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        payload = json.loads(raw)
        flattened_lines: list[str] = []
        _flatten_json(payload, flattened_lines, prefix="root", depth=0)
        normalized = "\n".join(flattened_lines)
    except json.JSONDecodeError:
        normalized = raw

    return _split_text_into_chunks(
        package_id=package_id,
        source_file=path.name,
        source_type="json",
        text=normalized,
        chunk_size=chunk_size,
        overlap=overlap,
        base_metadata={},
    )


def _flatten_json(value: Any, out: list[str], prefix: str, depth: int) -> None:
    if len(out) >= MAX_JSON_FLATTEN_LINES:
        return
    if depth > MAX_JSON_FLATTEN_DEPTH:
        out.append(f"{prefix}: [truncated: max nesting depth reached]")
        return

    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten_json(nested, out, f"{prefix}.{key}", depth + 1)
            if len(out) >= MAX_JSON_FLATTEN_LINES:
                return
        return

    if isinstance(value, list):
        for idx, nested in enumerate(value):
            _flatten_json(nested, out, f"{prefix}[{idx}]", depth + 1)
            if len(out) >= MAX_JSON_FLATTEN_LINES:
                return
        return

    serialized = str(value)
    if len(serialized) > MAX_JSON_VALUE_CHARS:
        serialized = serialized[:MAX_JSON_VALUE_CHARS] + "...[truncated]"
    out.append(f"{prefix}: {serialized}")


def _chunk_markdown_file(package_id: str, path: Path, chunk_size: int, overlap: int) -> list[TextChunk]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    sections = _split_markdown_sections(raw)
    chunks: list[TextChunk] = []

    for section_name, body in sections:
        base_metadata: dict[str, str] = {}
        if section_name:
            base_metadata["section"] = section_name

        section_chunks = _split_text_into_chunks(
            package_id=package_id,
            source_file=path.name,
            source_type="markdown",
            text=body,
            chunk_size=chunk_size,
            overlap=overlap,
            base_metadata=base_metadata,
        )
        chunks.extend(section_chunks)

    return chunks


def _split_markdown_sections(markdown_text: str) -> list[tuple[str, str]]:
    if not markdown_text.strip():
        return [("", "")]

    matches = list(SECTION_HEADING_PATTERN.finditer(markdown_text))
    if not matches:
        return [("", markdown_text)]

    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown_text)
        title = match.group("title").strip()
        body = markdown_text[start:end].strip()
        sections.append((title, body))

    return sections


def _split_text_into_chunks(
    package_id: str,
    source_file: str,
    source_type: str,
    text: str,
    chunk_size: int,
    overlap: int,
    base_metadata: dict[str, str],
) -> list[TextChunk]:
    cleaned = text.strip()
    if not cleaned:
        return []

    chunk_size = max(500, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))

    chunks: list[TextChunk] = []
    cursor = 0
    index = 0
    while cursor < len(cleaned):
        end = min(len(cleaned), cursor + chunk_size)
        snippet = cleaned[cursor:end]
        metadata = dict(base_metadata)

        page_match = PAGE_PATTERN.search(snippet)
        if page_match:
            metadata["page"] = page_match.group(1)

        chunks.append(
            TextChunk(
                package_id=package_id,
                source_file=source_file,
                source_type=source_type,
                content=snippet,
                chunk_index=index,
                metadata=metadata,
            )
        )

        if end >= len(cleaned):
            break
        cursor = end - overlap
        index += 1

    return chunks
