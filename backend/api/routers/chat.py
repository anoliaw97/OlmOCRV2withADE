from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.dependencies import get_runtime
from backend.schemas import ChatAskRequest, ChatAskResponse, CitationPayload
from core.llm_backends import DEFAULT_SYSTEM_PROMPT, LLMSettings


router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/ask", response_model=ChatAskResponse)
def chat_ask(request: ChatAskRequest) -> ChatAskResponse:
    runtime = get_runtime()
    settings_payload = request.llm_settings
    llm_settings = LLMSettings(
        backend=settings_payload.backend,
        model=settings_payload.model,
        system_prompt=settings_payload.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT,
        max_tokens=settings_payload.max_tokens,
        temperature=settings_payload.temperature,
        ollama_url=settings_payload.ollama_url,
        llama_cli_path=settings_payload.llama_cli_path,
    )

    try:
        response = runtime.ask(
            question=request.question,
            mode=request.mode,
            package_id=request.package_id,
            settings=llm_settings,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    citations = [
        CitationPayload(
            source_file=c.source_file,
            source_type=c.source_type,
            score=c.score,
            section=c.section,
            page=c.page,
        )
        for c in response.citations
    ]
    return ChatAskResponse(
        answer=response.answer,
        citations=citations,
        mode=response.mode,
        runtime=response.runtime,
        model=response.model,
    )
