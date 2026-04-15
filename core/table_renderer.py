from __future__ import annotations

import json
import re
from dataclasses import dataclass
from io import StringIO
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup


TABLE_HTML_PATTERN = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)


@dataclass(slots=True)
class ExtractedTable:
    title: str
    dataframe: pd.DataFrame | None
    source_type: str
    source_ref: str
    raw_text: str = ""


def extract_tables_from_markdown(markdown_text: str, source_ref: str) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    if not markdown_text.strip():
        return tables

    html_tables = TABLE_HTML_PATTERN.findall(markdown_text)
    for idx, html_block in enumerate(html_tables, start=1):
        tables.extend(_parse_html_table_block(html_block, source_ref, f"Markdown HTML table #{idx}"))

    markdown_tables = _extract_markdown_table_blocks(markdown_text)
    for idx, block in enumerate(markdown_tables, start=1):
        df = _markdown_block_to_dataframe(block)
        if df is None:
            continue
        tables.append(
            ExtractedTable(
                title=f"Markdown table #{idx}",
                dataframe=df,
                source_type="markdown",
                source_ref=source_ref,
                raw_text=block,
            )
        )

    return tables


def extract_tables_from_json_text(json_text: str, source_ref: str) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    if not json_text.strip():
        return tables

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        html_tables = TABLE_HTML_PATTERN.findall(json_text)
        for idx, html_block in enumerate(html_tables, start=1):
            tables.extend(_parse_html_table_block(html_block, source_ref, f"JSON embedded HTML table #{idx}"))
        return tables

    _walk_json_for_tables(payload, source_ref, tables, path="root")
    return tables


def _walk_json_for_tables(value: Any, source_ref: str, out: list[ExtractedTable], path: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _walk_json_for_tables(nested, source_ref, out, f"{path}.{key}")
        return

    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            df = pd.DataFrame(value)
            if not df.empty:
                out.append(
                    ExtractedTable(
                        title=f"JSON list table ({path})",
                        dataframe=df,
                        source_type="json",
                        source_ref=source_ref,
                        raw_text=df.to_csv(index=False),
                    )
                )
        for idx, nested in enumerate(value):
            _walk_json_for_tables(nested, source_ref, out, f"{path}[{idx}]")
        return

    if isinstance(value, str):
        html_tables = TABLE_HTML_PATTERN.findall(value)
        for idx, html_block in enumerate(html_tables, start=1):
            out.extend(_parse_html_table_block(html_block, source_ref, f"JSON HTML table ({path}) #{idx}"))


def _parse_html_table_block(html_block: str, source_ref: str, title: str) -> list[ExtractedTable]:
    soup = BeautifulSoup(html_block, "html.parser")
    found = soup.find_all("table")
    extracted: list[ExtractedTable] = []

    for table_index, table_tag in enumerate(found, start=1):
        rows: list[list[str]] = []
        for row_tag in table_tag.find_all("tr"):
            cells = row_tag.find_all(["th", "td"])
            row = [cell.get_text(" ", strip=True) for cell in cells]
            if row:
                rows.append(row)

        if not rows:
            extracted.append(
                ExtractedTable(
                    title=f"{title}.{table_index}",
                    dataframe=None,
                    source_type="html",
                    source_ref=source_ref,
                    raw_text=str(table_tag),
                )
            )
            continue

        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []
        if body and all(len(r) == len(header) for r in body):
            df = pd.DataFrame(body, columns=header)
        else:
            width = max(len(r) for r in rows)
            normalized_rows = [r + [""] * (width - len(r)) for r in rows]
            columns = [f"col_{i + 1}" for i in range(width)]
            df = pd.DataFrame(normalized_rows, columns=columns)

        extracted.append(
            ExtractedTable(
                title=f"{title}.{table_index}",
                dataframe=df,
                source_type="html",
                source_ref=source_ref,
                raw_text=str(table_tag),
            )
        )

    return extracted


def _extract_markdown_table_blocks(markdown_text: str) -> list[str]:
    lines = markdown_text.splitlines()
    blocks: list[str] = []
    i = 0

    while i < len(lines) - 1:
        line = lines[i].strip()
        next_line = lines[i + 1].strip()

        if "|" not in line or "|" not in next_line:
            i += 1
            continue

        if not re.match(r"^\|?\s*:?[-]{3,}.*\|", next_line):
            i += 1
            continue

        block_lines = [lines[i], lines[i + 1]]
        j = i + 2
        while j < len(lines):
            current = lines[j]
            if "|" not in current:
                break
            block_lines.append(current)
            j += 1

        blocks.append("\n".join(block_lines))
        i = j

    return blocks


def _markdown_block_to_dataframe(block: str) -> pd.DataFrame | None:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    header_cells = _split_md_row(lines[0])
    body_lines = lines[2:]
    rows = [_split_md_row(line) for line in body_lines]

    if not header_cells:
        return None

    width = max([len(header_cells)] + [len(r) for r in rows] or [len(header_cells)])
    header_cells = header_cells + [f"col_{i + 1}" for i in range(len(header_cells), width)]
    normalized_rows = [r + [""] * (width - len(r)) for r in rows]

    return pd.DataFrame(normalized_rows, columns=header_cells)


def _split_md_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    if not stripped:
        return []
    return [cell.strip() for cell in stripped.split("|")]
