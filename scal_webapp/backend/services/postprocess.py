from __future__ import annotations

from copy import deepcopy

import pandas as pd

from ..schemas import TableJSON


COLUMN_MAP = {
    "perm": "permeability_md",
    "k": "permeability_md",
    "permeability": "permeability_md",
    "poro": "porosity_pct",
    "phi": "porosity_pct",
    "porosity": "porosity_pct",
    "pc": "capillary_pressure_psi",
    "sw": "water_saturation_pct",
}


def normalize_name(col: str) -> str:
    c = col.strip().lower().replace(" ", "_").replace("-", "_")
    return COLUMN_MAP.get(c, c)


def normalize_tables(tables: list[TableJSON]) -> list[TableJSON]:
    out: list[TableJSON] = []
    for t in tables:
        tt = deepcopy(t)
        new_cols = [normalize_name(c) for c in tt.columns]

        # merge duplicate columns by first non-null value
        merged_rows = []
        for row in tt.rows:
            m = {}
            for old, val in row.items():
                key = normalize_name(old)
                if key not in m or m[key] in (None, ""):
                    m[key] = val
            merged_rows.append(m)

        # ensure all rows have all columns
        all_cols = sorted(set(new_cols) | {k for r in merged_rows for k in r.keys()})
        for r in merged_rows:
            for c in all_cols:
                r.setdefault(c, None)

        tt.columns = all_cols
        tt.rows = merged_rows
        tt.metadata["normalized"] = True
        out.append(tt)
    return out


def to_ml_ready_dataframe(tables: list[TableJSON]) -> pd.DataFrame:
    rows = []
    for t in tables:
        for r in t.rows:
            rr = {
                "file_name": t.file_name,
                "page_number": t.page_number,
                "table_id": t.table_id,
                "extraction_type": t.extraction_type,
                "table_title": t.table_title,
            }
            rr.update(r)
            rows.append(rr)
    df = pd.DataFrame(rows)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def build_rag_chunks(tables: list[TableJSON]) -> list[dict]:
    chunks: list[dict] = []
    for t in tables:
        for i, row in enumerate(t.rows, start=1):
            kv = "; ".join(f"{k}={row.get(k)}" for k in t.columns)
            text = f"{t.extraction_type} | {t.table_title or ''} | row {i}: {kv}"
            sample_id = str(row.get("sample_id") or row.get("sample") or "")
            chunks.append(
                {
                    "chunk_text": text,
                    "metadata": {
                        "file_name": t.file_name,
                        "page_number": t.page_number,
                        "table_id": t.table_id,
                        "extraction_type": t.extraction_type,
                        "sample_id": sample_id,
                        "report_name": t.metadata.get("report_name", ""),
                    },
                }
            )
    return chunks
