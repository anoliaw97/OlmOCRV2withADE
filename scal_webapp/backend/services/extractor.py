from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from ..schemas import TableJSON


SCAL_KEYWORDS = {
    "capillary_pressure": ["capillary", "pc", "sw", "drainage", "imbibition"],
    "relative_permeability": ["relative permeability", "krw", "kro", "krg"],
    "porosity_permeability": ["porosity", "permeability", "md", "phi"],
}


def _line_split(line: str) -> list[str]:
    # support comma, tab, and multi-space table style
    parts = re.split(r"\t|,|\s{2,}", line.strip())
    return [p.strip() for p in parts if p.strip()]


def _detect_type(text: str, default: str) -> str:
    t = text.lower()
    best_type = default
    best_score = 0
    for k, words in SCAL_KEYWORDS.items():
        score = sum(1 for w in words if w in t)
        if score > best_score:
            best_score = score
            best_type = k
    return best_type


def _parse_page_tables(page_text: str, file_name: str, page_number: int, allowed_types: list[str], default_use_case: str) -> list[TableJSON]:
    lines = [ln.rstrip() for ln in page_text.splitlines()]
    tables: list[TableJSON] = []
    table_idx = 0
    i = 0

    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue

        # Candidate table header by separators or obvious header words
        if ("|" in ln) or re.search(r"sample|depth|pressure|sw|krw|kro|porosity|permeability", ln.lower()):
            header = _line_split(ln.replace("|", "  "))
            if len(header) < 2:
                i += 1
                continue

            rows = []
            j = i + 1
            while j < len(lines):
                row_line = lines[j]
                if not row_line.strip():
                    break
                row = _line_split(row_line.replace("|", "  "))
                if len(row) >= 2:
                    rows.append(row)
                j += 1

            if len(rows) >= 2:
                table_idx += 1
                width = max(len(header), *(len(r) for r in rows))
                columns = header + [f"col_{c+1}" for c in range(len(header), width)]
                norm_rows = []
                for r in rows:
                    rr = r + [None] * (width - len(r))
                    norm_rows.append({columns[k]: rr[k] for k in range(width)})

                sample_text = "\n".join([ln] + [" ".join(r) for r in rows[:4]])
                etype = _detect_type(sample_text, default_use_case)
                if allowed_types and etype not in allowed_types:
                    i = j + 1
                    continue

                # units heuristic: columns with (%) / md / psi tokens
                units = {}
                for c in columns:
                    cl = c.lower()
                    if "%" in cl or "pct" in cl:
                        units[c] = "%"
                    elif "md" in cl:
                        units[c] = "md"
                    elif "psi" in cl:
                        units[c] = "psi"

                tables.append(
                    TableJSON(
                        file_name=file_name,
                        page_number=page_number,
                        table_id=f"T{page_number:03d}_{table_idx:02d}",
                        extraction_type=etype,
                        table_title=ln[:180],
                        columns=columns,
                        rows=norm_rows,
                        units=units or None,
                        metadata={
                            "report_name": Path(file_name).stem,
                            "parameter_type": etype,
                            "row_count": len(norm_rows),
                            "column_count": len(columns),
                        },
                    )
                )

                i = j
                continue

        i += 1

    return tables


def parse_page_range(page_count: int, page_range: str | None) -> list[int]:
    if not page_range:
        return list(range(1, page_count + 1))
    pages: set[int] = set()
    parts = [p.strip() for p in page_range.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            start, end = int(a), int(b)
            for idx in range(max(1, start), min(page_count, end) + 1):
                pages.add(idx)
        else:
            idx = int(p)
            if 1 <= idx <= page_count:
                pages.add(idx)
    return sorted(pages)


def extract_targeted_tables(pdf_path: str, default_use_case: str, allowed_types: list[str], page_range: str | None = None) -> list[TableJSON]:
    reader = PdfReader(pdf_path)
    selected_pages = parse_page_range(len(reader.pages), page_range)

    results: list[TableJSON] = []
    file_name = Path(pdf_path).name
    for page_num in selected_pages:
        text = reader.pages[page_num - 1].extract_text() or ""
        if not text.strip():
            continue
        tables = _parse_page_tables(text, file_name, page_num, allowed_types, default_use_case)
        results.extend(tables)
    return results
