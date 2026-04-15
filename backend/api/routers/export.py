from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.dependencies import get_runtime
from backend.schemas import ExportChatRequest, ExportChatResponse
from core.export_service import ChatRecord


router = APIRouter(prefix="/api/export", tags=["export"])


@router.post("/chat", response_model=ExportChatResponse)
def export_chat(request: ExportChatRequest) -> ExportChatResponse:
    runtime = get_runtime()
    records = [
        ChatRecord(
            timestamp=r.timestamp,
            mode=r.mode,
            runtime=r.runtime,
            model=r.model,
            question=r.question,
            answer=r.answer,
            citations=r.citations,
        )
        for r in request.records
    ]

    try:
        ok, message = runtime.export_records(request.destination, records)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}") from exc
    return ExportChatResponse(ok=ok, message=message)
