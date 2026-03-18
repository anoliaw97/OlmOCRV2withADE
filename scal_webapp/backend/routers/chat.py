from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import ProcessingLog
from ..schemas import QueryRequest
from ..services.indexer import LocalHybridIndex
from ..services.local_llm import get_rag_llm, get_usecase_llm
from ..services.prompt_library import list_prompts, save_prompt
from ..services.rag import synthesize_answer


router = APIRouter(prefix="/api/chat", tags=["chat"])

INDEX_DIR = Path("scal_webapp/data/index")


@router.post("/ask")
def ask_question(req: QueryRequest, db: Session = Depends(get_db)):
    # Important: chat uses indexed extracted JSON chunks only.
    filters = {
        "report_name": req.report_name,
        "file_name": req.file_name,
        "page_number": req.page_number,
        "extraction_type": req.extraction_type,
        "sample_id": req.sample_id,
    }

    index = LocalHybridIndex(INDEX_DIR)
    hits = index.search(req.question, top_k=req.top_k, filters=filters)
    try:
        response = synthesize_answer(req.question, hits, use_case_prompt=req.use_case_prompt)
    except Exception as e:
        response = {
            "answer": f"Local LLM answer generation failed: {e}",
            "sources": [
                {
                    "file_name": h["metadata"].get("file_name"),
                    "page_number": h["metadata"].get("page_number"),
                    "table_id": h["metadata"].get("table_id"),
                    "extraction_type": h["metadata"].get("extraction_type"),
                }
                for h in hits
            ],
        }

    db.add(
        ProcessingLog(
            report_id=None,
            stage="chat_retrieve",
            level="info",
            message=f"Query: {req.question}",
            payload_json={"filters": filters, "hits": len(hits)},
        )
    )
    db.commit()

    return response


@router.post("/load-llm")
def load_local_llms():
    try:
        rag = get_rag_llm()
        usecase = get_usecase_llm()
        rag.load()
        usecase.load()
        return {
            "ok": True,
            "rag_model": rag.model_name,
            "usecase_model": usecase.model_name,
            "loaded": True,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/logs")
def get_logs(limit: int = 200, db: Session = Depends(get_db)):
    rows = db.query(ProcessingLog).order_by(ProcessingLog.created_at.desc()).limit(limit).all()
    return [
        {
            "timestamp": r.created_at.isoformat(),
            "stage": r.stage,
            "level": r.level,
            "message": r.message,
            "payload": r.payload_json,
        }
        for r in rows
    ]


@router.delete("/logs")
def clear_logs(db: Session = Depends(get_db)):
    db.query(ProcessingLog).delete()
    db.commit()
    return {"ok": True}


@router.get("/prompts")
def get_saved_prompts():
    return {"prompts": list_prompts()}


@router.post("/prompts")
def save_chat_prompt(payload: dict):
    name = (payload or {}).get("name", "").strip()
    text = (payload or {}).get("text", "").strip()
    if not name or not text:
        return {"ok": False, "error": "Both name and text are required"}
    prompts = save_prompt(name, text)
    return {"ok": True, "prompts": prompts}
