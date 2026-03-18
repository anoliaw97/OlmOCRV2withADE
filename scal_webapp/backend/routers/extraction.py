from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import ExtractedTable, RagChunk, Report
from ..schemas import ExtractionSettings, TableJSON
from ..services.exporter import export_excel, export_json, export_word
from ..services.extractor import extract_targeted_tables, parse_page_range
from ..services.indexer import LocalHybridIndex
from ..services.logger import log_event
from ..services.postprocess import build_rag_chunks, normalize_tables
from ..services.web_olmocr_runtime import default_olmocr_prompt, get_vlm
from olmocr.data.renderpdf import render_pdf_to_base64png


router = APIRouter(prefix="/api/extraction", tags=["extraction"])

DATA_DIR = Path("scal_webapp/data")
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORT_DIR = DATA_DIR / "exports"
INDEX_DIR = DATA_DIR / "index"
for d in (UPLOAD_DIR, EXPORT_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)


@router.post("/run")
async def run_extraction(
    file: UploadFile = File(...),
    settings_json: str = Form("{}"),
    db: Session = Depends(get_db),
):
    settings = ExtractionSettings(**json.loads(settings_json or "{}"))
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF is supported")

    dest = UPLOAD_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)

    report = Report(file_name=file.filename, report_name=Path(file.filename).stem, status="processing")
    db.add(report)
    db.commit()
    db.refresh(report)

    log_event(db, "ingest", f"Uploaded {file.filename}", report.id)

    # Desktop-like behavior:
    # - Layman mode default: full extraction via loaded VLM + default olmOCR prompt
    # - Operator mode: can run targeted parser flow
    if settings.mode == "operator" and settings.extraction_types and settings.model_name == "offline_heuristic":
        tables = extract_targeted_tables(
            pdf_path=str(dest),
            default_use_case=settings.use_case,
            allowed_types=settings.extraction_types,
            page_range=settings.page_range,
        )
        log_event(db, "extract", f"Operator targeted extraction produced {len(tables)} table JSON records", report.id)
    else:
        # VLM extraction path (default prompt unless user overrides)
        vlm = get_vlm()
        if not vlm.loaded:
            raise HTTPException(status_code=400, detail="VLM is not loaded. Click 'Load VLM' first.")

        from pypdf import PdfReader

        reader = PdfReader(str(dest))
        pages = parse_page_range(len(reader.pages), settings.page_range)
        use_prompt = settings.prompt_text or default_olmocr_prompt()

        page_tables: list[dict] = []
        for page_num in pages:
            res = vlm.extract_page(str(dest), page_num, prompt=use_prompt)
            page_tables.append(
                {
                    "file_name": file.filename,
                    "page_number": page_num,
                    "table_id": f"P{page_num:03d}_FULL",
                    "extraction_type": "full_page_text",
                    "table_title": f"olmOCR full extraction page {page_num}",
                    "columns": ["page_text"],
                    "rows": [{"page_text": res.get("raw_response", "")}],
                    "units": {},
                    "metadata": {
                        "report_name": Path(file.filename).stem,
                        "prompt_used": res.get("prompt_used", use_prompt),
                        "input_tokens": res.get("input_tokens", 0),
                        "output_tokens": res.get("output_tokens", 0),
                        "total_tokens": res.get("total_tokens", 0),
                    },
                }
            )

        tables = [TableJSON(**t) for t in page_tables]
        log_event(db, "extract", f"Default VLM full extraction produced {len(tables)} page JSON records", report.id)

    if settings.normalize:
        tables = normalize_tables(tables)
        log_event(db, "postprocess", "Column normalization and duplicate merge complete", report.id)

    # persist table JSON records
    for t in tables:
        db.add(
            ExtractedTable(
                report_id=report.id,
                file_name=t.file_name,
                page_number=t.page_number,
                table_id=t.table_id,
                extraction_type=t.extraction_type,
                table_title=t.table_title,
                columns_json=t.columns,
                rows_json=t.rows,
                units_json=t.units,
                metadata_json=t.metadata,
            )
        )
    db.commit()

    # rag chunks
    rag_chunks = build_rag_chunks(tables)
    for c in rag_chunks:
        m = c["metadata"]
        db.add(
            RagChunk(
                report_id=report.id,
                table_ref_id=0,
                chunk_text=c["chunk_text"],
                metadata_json=m,
            )
        )
    db.commit()
    log_event(db, "index_prepare", f"Prepared {len(rag_chunks)} RAG chunks", report.id)

    # local hybrid index
    if settings.build_index:
        index = LocalHybridIndex(INDEX_DIR)
        index.build(rag_chunks)
        log_event(db, "index", "Local hybrid index rebuilt", report.id)

    report.status = "ready"
    db.commit()

    total_input_tokens = 0
    total_output_tokens = 0
    for t in tables:
        meta = getattr(t, "metadata", {}) or {}
        total_input_tokens += int(meta.get("input_tokens", 0) or 0)
        total_output_tokens += int(meta.get("output_tokens", 0) or 0)

    return {
        "report_id": report.id,
        "tables": len(tables),
        "rag_chunks": len(rag_chunks),
        "pages_extracted": len(tables),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "status": "ready",
    }


@router.post("/import-json")
async def import_existing_json(
    json_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not json_file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")

    raw = await json_file.read()
    try:
        payload = json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    report = Report(file_name=json_file.filename, report_name=Path(json_file.filename).stem, status="processing")
    db.add(report)
    db.commit()
    db.refresh(report)

    log_event(db, "ingest", f"Imported existing extraction JSON {json_file.filename}", report.id)

    tables: list[dict] = []
    if isinstance(payload, list):
        # Case A: already table-json list
        if payload and isinstance(payload[0], dict) and ("rows" in payload[0] or "columns" in payload[0]):
            tables = payload
        # Case B: page extraction list from GUI with raw_response
        elif payload and isinstance(payload[0], dict) and "raw_response" in payload[0]:
            temp = []
            for i, p in enumerate(payload, start=1):
                page_number = int(p.get("page", i) or i)
                text = str(p.get("raw_response", ""))
                temp.append(
                    {
                        "file_name": p.get("source_file", json_file.filename),
                        "page_number": page_number,
                        "table_id": f"P{page_number:03d}_FULL",
                        "extraction_type": "full_page_text",
                        "table_title": f"Imported full page {page_number}",
                        "columns": ["page_text"],
                        "rows": [{"page_text": text}],
                        "units": {},
                        "metadata": {
                            "report_name": Path(json_file.filename).stem,
                            "imported": True,
                            "source": "raw_response_list",
                        },
                    }
                )
            tables = temp
    elif isinstance(payload, dict):
        if isinstance(payload.get("tables"), list):
            tables = payload["tables"]
        elif isinstance(payload.get("records"), list):
            # fallback map records into single pseudo table
            tables = [
                {
                    "file_name": json_file.filename,
                    "page_number": 1,
                    "table_id": "IMPORTED_001",
                    "extraction_type": "imported_json",
                    "table_title": "Imported structured records",
                    "columns": sorted({k for r in payload["records"] if isinstance(r, dict) for k in r.keys()}),
                    "rows": payload["records"],
                    "units": {},
                    "metadata": {"report_name": Path(json_file.filename).stem, "imported": True},
                }
            ]

    # normalize structure and persist
    normalized_tables = []
    for i, t in enumerate(tables, start=1):
        normalized_tables.append(
            {
                "file_name": t.get("file_name", json_file.filename),
                "page_number": int(t.get("page_number", 1) or 1),
                "table_id": t.get("table_id", f"IMPORTED_{i:03d}"),
                "extraction_type": t.get("extraction_type", "imported_json"),
                "table_title": t.get("table_title"),
                "columns": t.get("columns", []),
                "rows": t.get("rows", []),
                "units": t.get("units", {}),
                "metadata": t.get("metadata", {"report_name": Path(json_file.filename).stem, "imported": True}),
            }
        )

    for t in normalized_tables:
        db.add(
            ExtractedTable(
                report_id=report.id,
                file_name=t["file_name"],
                page_number=t["page_number"],
                table_id=t["table_id"],
                extraction_type=t["extraction_type"],
                table_title=t["table_title"],
                columns_json=t["columns"],
                rows_json=t["rows"],
                units_json=t["units"],
                metadata_json=t["metadata"],
            )
        )
    db.commit()

    # rebuild index from all report tables (simple deterministic behavior)
    rows = db.query(ExtractedTable).all()
    all_tables = [
        {
            "file_name": r.file_name,
            "page_number": r.page_number,
            "table_id": r.table_id,
            "extraction_type": r.extraction_type,
            "table_title": r.table_title,
            "columns": r.columns_json,
            "rows": r.rows_json,
            "units": r.units_json,
            "metadata": r.metadata_json,
        }
        for r in rows
    ]

    rag_chunks = []
    for t in all_tables:
        for idx, row in enumerate(t.get("rows", []), start=1):
            txt = "; ".join(f"{k}={row.get(k)}" for k in t.get("columns", []))
            rag_chunks.append(
                {
                    "chunk_text": f"{t.get('extraction_type')} | {t.get('table_title') or ''} | row {idx}: {txt}",
                    "metadata": {
                        "file_name": t.get("file_name"),
                        "page_number": t.get("page_number"),
                        "table_id": t.get("table_id"),
                        "extraction_type": t.get("extraction_type"),
                        "sample_id": str(row.get("sample_id") or row.get("sample") or ""),
                        "report_name": (t.get("metadata") or {}).get("report_name", ""),
                    },
                }
            )

    index = LocalHybridIndex(INDEX_DIR)
    if rag_chunks:
        index.build(rag_chunks)
    log_event(db, "index", f"Index rebuilt from imported JSON ({len(rag_chunks)} chunks)", report.id)

    report.status = "ready"
    db.commit()
    return {"report_id": report.id, "imported_tables": len(normalized_tables), "rag_chunks": len(rag_chunks), "status": "ready"}


@router.get("/reports")
def list_reports(db: Session = Depends(get_db)):
    rows = db.query(Report).order_by(Report.uploaded_at.desc()).all()
    return [
        {
            "id": r.id,
            "file_name": r.file_name,
            "report_name": r.report_name,
            "status": r.status,
            "uploaded_at": r.uploaded_at.isoformat(),
        }
        for r in rows
    ]


@router.get("/report/{report_id}/tables")
def get_report_tables(report_id: int, db: Session = Depends(get_db)):
    rows = db.query(ExtractedTable).filter(ExtractedTable.report_id == report_id).all()
    return [
        {
            "file_name": r.file_name,
            "page_number": r.page_number,
            "table_id": r.table_id,
            "extraction_type": r.extraction_type,
            "table_title": r.table_title,
            "columns": r.columns_json,
            "rows": r.rows_json,
            "units": r.units_json,
            "metadata": r.metadata_json,
        }
        for r in rows
    ]


@router.get("/report/{report_id}/pages")
def get_report_pages(report_id: int, db: Session = Depends(get_db)):
    rows = (
        db.query(ExtractedTable)
        .filter(ExtractedTable.report_id == report_id)
        .order_by(ExtractedTable.page_number.asc())
        .all()
    )
    pages = []
    for r in rows:
        text = ""
        if isinstance(r.rows_json, list) and r.rows_json and isinstance(r.rows_json[0], dict):
            text = str(r.rows_json[0].get("page_text", ""))
        meta = r.metadata_json or {}
        pages.append(
            {
                "page_number": r.page_number,
                "table_id": r.table_id,
                "extraction_type": r.extraction_type,
                "raw_response": text,
                "input_tokens": int(meta.get("input_tokens", 0) or 0),
                "output_tokens": int(meta.get("output_tokens", 0) or 0),
                "total_tokens": int(meta.get("total_tokens", 0) or 0),
            }
        )
    return {"report_id": report_id, "pages": pages}


@router.get("/report/{report_id}/thumbnail/{page_number}")
def get_page_thumbnail(report_id: int, page_number: int, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    pdf_path = UPLOAD_DIR / report.file_name
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Source PDF not found in uploads")

    try:
        img_b64 = render_pdf_to_base64png(str(pdf_path), page_number, target_longest_image_dim=360)
        return {
            "report_id": report_id,
            "page_number": page_number,
            "image_data_url": f"data:image/png;base64,{img_b64}",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Thumbnail render failed: {e}")


@router.get("/report/{report_id}/export/{fmt}")
def export_report(report_id: int, fmt: str, db: Session = Depends(get_db)):
    rows = db.query(ExtractedTable).filter(ExtractedTable.report_id == report_id).all()
    if not rows:
        raise HTTPException(status_code=404, detail="No tables for report")

    tables = [
        {
            "file_name": r.file_name,
            "page_number": r.page_number,
            "table_id": r.table_id,
            "extraction_type": r.extraction_type,
            "table_title": r.table_title,
            "columns": r.columns_json,
            "rows": r.rows_json,
            "units": r.units_json,
            "metadata": r.metadata_json,
        }
        for r in rows
    ]
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if fmt == "json":
        out = EXPORT_DIR / f"report_{report_id}_{stamp}.json"
        export_json(out, {"report_id": report_id, "tables": tables})
        return FileResponse(out)

    ml_rows = []
    for t in tables:
        for r in t["rows"]:
            rr = {
                "file_name": t["file_name"],
                "page_number": t["page_number"],
                "table_id": t["table_id"],
                "extraction_type": t["extraction_type"],
                "table_title": t.get("table_title"),
            }
            rr.update(r)
            ml_rows.append(rr)
    ml_df = pd.DataFrame(ml_rows)

    if fmt == "xlsx":
        out = EXPORT_DIR / f"report_{report_id}_{stamp}.xlsx"
        export_excel(out, tables, ml_df)
        return FileResponse(out)

    if fmt == "docx":
        out = EXPORT_DIR / f"report_{report_id}_{stamp}.docx"
        export_word(out, tables)
        return FileResponse(out)

    raise HTTPException(status_code=400, detail="Unsupported format")
