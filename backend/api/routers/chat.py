from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.dependencies import get_runtime
from backend.schemas import (
    ChatAskRequest,
    ChatAskResponse,
    ChatMetricsPayload,
    ChatSessionCreateRequest,
    ChatSessionPayload,
    ChatSessionResponse,
    ChatSessionsResponse,
    CitationPayload,
    GenericOkResponse,
    SessionMessagePayload,
)
from core.llm_backends import DEFAULT_SYSTEM_PROMPT, LLMSettings
from core.model_registry import detect_model_context_limit
from core.export_intent_parser import parse_export_intent
from core.query_router import QueryRouter


router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.get("/sessions", response_model=ChatSessionsResponse)
def chat_sessions() -> ChatSessionsResponse:
    runtime = get_runtime()
    sessions = runtime.list_sessions()
    runtime.log("debug", f"Session list requested; {len(sessions)} session(s) available.")
    return ChatSessionsResponse(sessions=sessions)


@router.post("/session/new", response_model=ChatSessionResponse)
def chat_session_new(request: ChatSessionCreateRequest) -> ChatSessionResponse:
    runtime = get_runtime()
    session = runtime.create_session(request.title)
    runtime.log("status", f"Created chat session: {session.get('session_id', '')}")
    return ChatSessionResponse(session=_to_session_payload(session))


@router.get("/session/{session_id}", response_model=ChatSessionResponse)
def chat_session_get(session_id: str) -> ChatSessionResponse:
    runtime = get_runtime()
    session = runtime.get_session(session_id)
    if not session:
        runtime.log("error", f"Session lookup failed: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    runtime.log("debug", f"Session opened: {session_id}")
    return ChatSessionResponse(session=_to_session_payload(session))


@router.delete("/session/{session_id}", response_model=GenericOkResponse)
def chat_session_delete(session_id: str) -> GenericOkResponse:
    runtime = get_runtime()
    deleted = runtime.delete_session(session_id)
    if not deleted:
        runtime.log("error", f"Session delete failed; not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    runtime.log("status", f"Deleted chat session: {session_id}")
    return GenericOkResponse(ok=True)


@router.post("/ask", response_model=ChatAskResponse)
def chat_ask(request: ChatAskRequest) -> ChatAskResponse:
    runtime = get_runtime()
    runtime.log(
        "debug",
        f"Chat request received: mode={request.mode}, package={request.package_id or '-'}, backend={request.llm_settings.backend}",
    )
    settings_payload = request.llm_settings
    llm_settings = LLMSettings(
        backend=settings_payload.backend,
        model=settings_payload.model,
        system_prompt=settings_payload.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT,
        max_tokens=settings_payload.max_tokens,
        temperature=settings_payload.temperature,
        ollama_url=settings_payload.ollama_url,
        llama_cli_path=settings_payload.llama_cli_path,
        context_limit=settings_payload.context_limit,
    )

    session_id = (request.session_id or "").strip()
    if not session_id:
        session_id = str(runtime.create_session("Workflow Chat").get("session_id") or "")

    session_history = runtime.recent_session_messages(session_id=session_id, limit=8)
    package = runtime.get_package(request.package_id)

    export_intent = parse_export_intent(request.question)
    if export_intent.is_export:
        runtime.log(
            "status",
            f"Export chat intent detected: format={export_intent.export_format}, confidence={export_intent.confidence:.2f}",
        )
        export_result = runtime.chat_export_agent.run_export(
            question=request.question,
            export_format=export_intent.export_format,
            package=package,
            package_id=request.package_id,
        )

        if export_result.ok:
            answer = (
                f"I found {export_result.matched_chunks} relevant extracted chunk(s) and generated a "
                f"{export_result.export_format.upper()} file.\n"
                f"Saved at: {export_result.file_path}"
            )
            runtime.log("status", answer)
            return ChatAskResponse(
                answer=answer,
                citations=[],
                mode=request.mode,
                runtime="assistant",
                model=settings_payload.model,
                assistant_name=_assistant_name(settings_payload.model, "assistant"),
                session_id=session_id,
                reasoning_chain=[
                    "Detected export intent from user prompt.",
                    "Retrieved relevant extracted content.",
                    f"Generated {export_result.export_format.upper()} export file.",
                ],
                metrics=ChatMetricsPayload(),
                route_type="document",
                route_confidence=float(export_intent.confidence),
                route_reason=export_intent.reason,
                action_type=f"export_{export_result.export_format}",
                export_file_path=export_result.file_path,
                export_format=export_result.export_format,
            )

        runtime.log("error", f"Export action failed: {export_result.message}")
        return ChatAskResponse(
            answer=export_result.message,
            citations=[],
            mode=request.mode,
            runtime="assistant",
            model=settings_payload.model,
            assistant_name=_assistant_name(settings_payload.model, "assistant"),
            session_id=session_id,
            reasoning_chain=[
                "Detected export intent from user prompt.",
                "Tried to gather extracted content for export.",
                "Export generation could not complete with available content.",
            ],
            metrics=ChatMetricsPayload(),
            route_type="document",
            route_confidence=float(export_intent.confidence),
            route_reason=export_intent.reason,
            action_type="export_failed",
            export_file_path=export_result.file_path,
            export_format=export_result.export_format,
        )

    router = QueryRouter(runtime.retrieval_engine)
    route = router.route(
        question=request.question,
        package=package,
        preferred_mode=request.mode,
        package_id=request.package_id,
    )
    runtime.log(
        "debug",
        f"Query route decision: type={route.type}, confidence={route.confidence:.2f}, reason={route.reason}",
    )

    metadata_answer = _runtime_metadata_answer(request.question, runtime)
    if metadata_answer is not None:
        return ChatAskResponse(
            answer=metadata_answer,
            citations=[],
            mode=request.mode,
            runtime="assistant",
            model=settings_payload.model,
            assistant_name=_assistant_name(settings_payload.model, "assistant"),
            session_id=session_id,
            reasoning_chain=["Answered from runtime metadata without retrieval."],
            metrics=ChatMetricsPayload(),
            route_type="general",
            route_confidence=0.99,
            route_reason="runtime-metadata-question",
            action_type="chat",
        )

    ml_hint = runtime.chat_agent.maybe_handle_ml_command(request.question, runtime.packages)
    if ml_hint is not None:
        return ChatAskResponse(
            answer=ml_hint,
            citations=[],
            mode=request.mode,
            runtime="assistant",
            model=settings_payload.model,
            assistant_name=_assistant_name(settings_payload.model, "assistant"),
            session_id=session_id,
            reasoning_chain=["Detected ML workflow intent and returned operation guidance."],
            metrics=ChatMetricsPayload(),
            route_type="general",
            route_confidence=0.93,
            route_reason="ml-workflow-intent",
            action_type="chat",
        )

    started = time.perf_counter()
    try:
        response = runtime.ask(
            question=request.question,
            mode=request.mode,
            package_id=request.package_id,
            settings=llm_settings,
            session_history=session_history,
            route_decision=route,
        )
    except Exception as exc:
        runtime.log("error", f"Chat failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc
    total_ms = (time.perf_counter() - started) * 1000.0

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

    backend_for_context = _normalize_backend_name(response.runtime, settings_payload.backend)
    model_for_context = (response.model or settings_payload.model or "").strip()
    context_limit, context_source = detect_model_context_limit(
        backend=backend_for_context,
        model_name=model_for_context,
        ollama_url=settings_payload.ollama_url,
    )
    if int(context_limit or 0) <= 0:
        context_limit = max(0, int(settings_payload.context_limit or 0))
        if context_limit > 0:
            context_source = "configured"

    assistant_name = _assistant_name(response.model, response.runtime)
    citation_text = "; ".join(
        f"{item.source_file}:{item.source_type}:{item.score:.2f}" for item in response.citations
    )

    try:
        runtime.append_session_messages(
            session_id,
            [
                {
                    "role": "user",
                    "content": request.question,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "runtime": "",
                    "model": "",
                    "citations": "",
                    "reasoning_chain": [],
                },
                {
                    "role": "assistant",
                    "content": response.answer,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "runtime": response.runtime,
                    "model": response.model,
                    "citations": citation_text,
                    "reasoning_chain": response.reasoning_chain,
                },
            ],
        )
    except Exception:
        pass

    metrics = ChatMetricsPayload(
        context_limit=int(context_limit),
        context_limit_source=context_source,
        context_chars=int(response.context_chars),
        context_truncated=bool(response.context_truncated),
        retrieval_chunks=int(response.retrieval_chunks),
        retrieval_ms=round(float(response.retrieval_ms), 2),
        generation_ms=round(float(response.generation_ms), 2),
        total_ms=round(float(total_ms), 2),
    )

    runtime.log(
        "status",
        f"Chat answered with runtime={response.runtime}, model={response.model or '-'}, total_ms={metrics.total_ms}",
    )
    runtime.log(
        "reasoning",
        (
            f"Retrieved chunks={metrics.retrieval_chunks}, context={metrics.context_chars}, "
            f"truncated={metrics.context_truncated}, retrieval_ms={metrics.retrieval_ms}, "
            f"generation_ms={metrics.generation_ms}"
        ),
    )
    for line in response.reasoning_chain[:12]:
        runtime.log("reasoning", str(line))

    return ChatAskResponse(
        answer=response.answer,
        citations=citations,
        mode=response.mode,
        runtime=response.runtime,
        model=response.model,
        assistant_name=assistant_name,
        session_id=session_id,
        reasoning_chain=response.reasoning_chain,
        metrics=metrics,
        route_type=response.route_type,
        route_confidence=float(response.route_confidence),
        route_reason=response.route_reason,
        action_type="chat",
        export_file_path="",
        export_format="",
    )


def _normalize_backend_name(runtime_name: str, configured_backend: str) -> str:
    runtime_norm = (runtime_name or "").strip().lower()
    if runtime_norm == "llama.cpp":
        return "llamacpp"
    if runtime_norm in {"ollama", "transformers", "heuristic"}:
        return runtime_norm
    if runtime_norm in {"none", "fallback-heuristic"}:
        cfg = (configured_backend or "").strip().lower()
        if cfg == "llama.cpp":
            return "llamacpp"
        if cfg:
            return cfg
        return "heuristic"
    cfg = (configured_backend or "").strip().lower()
    return "llamacpp" if cfg == "llama.cpp" else (cfg or "heuristic")


def _assistant_name(model_name: str, runtime_name: str) -> str:
    model = (model_name or "").strip()
    if model and model.lower() != "none":
        path_like = Path(model)
        if path_like.suffix:
            return path_like.name
        return model

    runtime_norm = (runtime_name or "").strip().lower()
    if runtime_norm in {"heuristic", "fallback-heuristic"}:
        return "Heuristic Assistant"
    if runtime_norm == "llama.cpp":
        return "llama.cpp"
    if runtime_norm:
        return runtime_name
    return "Assistant"


def _to_session_payload(session: dict) -> ChatSessionPayload:
    messages = []
    for message in session.get("messages", []):
        if not isinstance(message, dict):
            continue
        messages.append(
            SessionMessagePayload(
                role=str(message.get("role") or "assistant"),
                content=str(message.get("content") or ""),
                time=str(message.get("time") or ""),
                runtime=str(message.get("runtime") or ""),
                model=str(message.get("model") or ""),
                citations=str(message.get("citations") or ""),
                reasoning_chain=[str(item) for item in message.get("reasoning_chain", [])],
            )
        )

    return ChatSessionPayload(
        session_id=str(session.get("session_id") or ""),
        title=str(session.get("title") or "Workflow Chat"),
        created_at=str(session.get("created_at") or ""),
        updated_at=str(session.get("updated_at") or ""),
        messages=messages,
    )


def _runtime_metadata_answer(question: str, runtime) -> str | None:
    q = " ".join(str(question or "").lower().split())
    if "how many reports" in q and "database" in q:
        loaded = len(runtime.packages)
        return f"You currently have {loaded} report package(s) loaded in this session."
    if "how many" in q and "package" in q:
        loaded = len(runtime.packages)
        return f"There are {loaded} loaded package(s) right now."
    return None
