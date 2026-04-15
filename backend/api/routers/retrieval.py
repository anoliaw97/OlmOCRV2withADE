from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.dependencies import get_runtime
from backend.schemas import BuildIndexResponse, RetrievalChunkPayload, RetrievalQueryRequest, RetrievalQueryResponse


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


@router.post("/query", response_model=RetrievalQueryResponse)
def retrieval_query(request: RetrievalQueryRequest) -> RetrievalQueryResponse:
    runtime = get_runtime()
    runtime.log("debug", f"Retrieval query received: mode={request.mode} top_k={request.top_k}")
    try:
        chunks = runtime.retrieve(
            question=request.question,
            mode=request.mode,
            package_id=request.package_id,
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
