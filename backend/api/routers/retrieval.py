from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.dependencies import get_runtime
from backend.schemas import (
    BuildIndexResponse,
    GenericMessageResponse,
    IndexStatusResponse,
    RetrievalChunkPayload,
    RetrievalQueryRequest,
    RetrievalQueryResponse,
)


router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@router.post("/index/build", response_model=BuildIndexResponse)
def build_index() -> BuildIndexResponse:
    runtime = get_runtime()
    if not runtime.packages:
        runtime.log("error", "Index build requested without loaded packages.")
        raise HTTPException(status_code=400, detail="No packages loaded.")
    runtime.log("status", f"RAG index build requested for {len(runtime.packages)} package(s).")
    try:
        chunk_count = runtime.build_index()
    except Exception as exc:
        runtime.log("error", f"RAG index build failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to build index: {exc}") from exc
    runtime.log("status", f"RAG index build complete: {chunk_count} chunk(s).")
    return BuildIndexResponse(indexed_chunks=chunk_count, package_count=len(runtime.packages))


@router.post("/index/update", response_model=BuildIndexResponse)
def update_index() -> BuildIndexResponse:
    runtime = get_runtime()
    if not runtime.packages:
        runtime.log("error", "Index update requested without loaded packages.")
        raise HTTPException(status_code=400, detail="No packages loaded.")
    runtime.log("status", f"RAG index update requested for {len(runtime.packages)} package(s).")
    try:
        chunk_count = runtime.update_index()
    except Exception as exc:
        runtime.log("error", f"RAG index update failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to update index: {exc}") from exc
    runtime.log("status", f"RAG index update complete: {chunk_count} chunk(s).")
    return BuildIndexResponse(indexed_chunks=chunk_count, package_count=len(runtime.packages))


@router.post("/index/clear", response_model=GenericMessageResponse)
def clear_index() -> GenericMessageResponse:
    runtime = get_runtime()
    try:
        runtime.clear_index()
    except Exception as exc:
        runtime.log("error", f"RAG index clear failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {exc}") from exc
    runtime.log("status", "RAG index cleared.")
    return GenericMessageResponse(ok=True, message="RAG index cleared.")


@router.get("/index/status", response_model=IndexStatusResponse)
def index_status() -> IndexStatusResponse:
    runtime = get_runtime()
    status = runtime.rag_status()
    return IndexStatusResponse(
        ready=bool(status.get("ready", False)),
        indexed_chunks=int(status.get("chunk_count", 0)),
        indexed_packages=int(status.get("package_count", 0)),
        last_updated=str(status.get("last_updated") or ""),
    )


@router.post("/query", response_model=RetrievalQueryResponse)
def retrieval_query(request: RetrievalQueryRequest) -> RetrievalQueryResponse:
    runtime = get_runtime()
    runtime.log("debug", f"Retrieval query received: mode={request.mode} top_k={request.top_k}")
    try:
        chunks = runtime.retrieve(
            question=request.question,
            mode=request.mode,
            package_id=request.package_id if request.mode == "direct" else None,
            top_k=max(1, min(int(request.top_k), 12)),
        )
    except Exception as exc:
        runtime.log("error", f"Retrieval query failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    runtime.log("debug", f"Retrieval query returned {len(chunks)} chunk(s).")

    payload = [
        RetrievalChunkPayload(
            package_id=c.package_id,
            source_file=c.source_file,
            source_type=c.source_type,
            content=c.content,
            score=c.score,
            section=c.section,
            page=c.page,
            table_name=c.table_name,
        )
        for c in chunks
    ]
    return RetrievalQueryResponse(chunks=payload)
