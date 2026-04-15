from __future__ import annotations

import re
import time
from dataclasses import dataclass

from core.llm_backends import LLMBackendError, LLMOrchestrator, LLMSettings
from core.loaders import DocumentPackage
from core.query_router import QueryRouter, RouteDecision
from core.retriever import RetrievalEngine, RetrievedChunk


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(slots=True)
class Citation:
    source_file: str
    source_type: str
    score: float
    section: str = ""
    page: str = ""


@dataclass(slots=True)
class ChatResponse:
    answer: str
    citations: list[Citation]
    mode: str
    runtime: str
    model: str
    reasoning_chain: list[str]
    context_chars: int
    context_truncated: bool
    retrieval_chunks: int
    retrieval_ms: float
    generation_ms: float
    route_type: str = "general"
    route_confidence: float = 0.0
    route_reason: str = ""


class ChatAgent:
    """Grounded chat over extracted JSON/Markdown/TXT with pluggable LLM backends."""

    def __init__(self, retrieval_engine: RetrievalEngine, llm_orchestrator: LLMOrchestrator | None = None) -> None:
        self.retrieval_engine = retrieval_engine
        self.llm_orchestrator = llm_orchestrator or LLMOrchestrator()
        self.query_router = QueryRouter(retrieval_engine)

    def ask(
        self,
        question: str,
        package: DocumentPackage | None,
        mode: str,
        llm_settings: LLMSettings,
        session_history: list[dict] | None = None,
        route_decision: RouteDecision | None = None,
        package_id: str | None = None,
    ) -> ChatResponse:
        cleaned = question.strip()
        retrieval_ms = 0.0
        generation_ms = 0.0
        history_text = _build_history_context(session_history or [])
        route = route_decision or self.query_router.route(
            question=cleaned,
            package=package,
            preferred_mode=mode,
            package_id=package_id,
        )
        route_type = route.type
        if not cleaned:
            return ChatResponse(
                answer="Please enter a question.",
                citations=[],
                mode=mode,
                runtime="none",
                model="",
                reasoning_chain=["Question is empty."],
                context_chars=0,
                context_truncated=False,
                retrieval_chunks=0,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                route_type=route_type,
                route_confidence=route.confidence,
                route_reason=route.reason,
            )

        reporting_answer = self._handle_reporting_question(cleaned, package)
        if reporting_answer is not None:
            return ChatResponse(
                answer=reporting_answer,
                citations=[],
                mode=mode,
                runtime="assistant",
                model=llm_settings.model,
                reasoning_chain=["Handled as runtime metadata question without retrieval."],
                context_chars=0,
                context_truncated=False,
                retrieval_chunks=0,
                retrieval_ms=0.0,
                generation_ms=0.0,
                route_type="general",
                route_confidence=0.99,
                route_reason="runtime-metadata-question",
            )
        retrieval_started = time.perf_counter()
        retrieved: list[RetrievedChunk] = []
        if route_type in {"document", "hybrid"}:
            if mode == "rag":
                retrieved = self.retrieval_engine.retrieve_rag(
                    cleaned,
                    top_k=6,
                    package_id=package_id,
                    min_score=0.55,
                    allow_fallback=True,
                )
                if not retrieved and package is not None:
                    retrieved = self.retrieval_engine.retrieve_direct(
                        package,
                        cleaned,
                        top_k=6,
                        min_score=0.5,
                        allow_fallback=True,
                    )
            else:
                if package is not None:
                    retrieved = self.retrieval_engine.retrieve_direct(
                        package,
                        cleaned,
                        top_k=6,
                        min_score=0.5,
                        allow_fallback=True,
                    )
                if not retrieved:
                    retrieved = self.retrieval_engine.retrieve_rag(
                        cleaned,
                        top_k=6,
                        package_id=package_id,
                        min_score=0.55,
                        allow_fallback=True,
                    )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
        top_retrieval_score = max((float(chunk.score) for chunk in retrieved), default=0.0)

        if not retrieved:
            if route_type == "document":
                route_type = "hybrid"
        elif route_type == "document" and route.confidence < 0.75 and top_retrieval_score < 1.1:
            route_type = "hybrid"

        citations = [
            Citation(
                source_file=chunk.source_file,
                source_type=chunk.source_type,
                score=chunk.score,
                section=chunk.section,
                page=chunk.page,
            )
            for chunk in retrieved[:6]
        ]

        context_limit = max(2000, int(llm_settings.context_limit or 24000))
        context, context_truncated = self._build_context(
            retrieved,
            max_chars=context_limit,
            history_text=history_text,
        )
        context_chars = len(context)

        reasoning_chain = [
            f"Mode: {mode}.",
            f"Route: {route_type} ({route.confidence:.2f}, reason={route.reason}).",
            f"Retrieved {len(retrieved)} chunk(s) from extracted JSON/MD/TXT.",
            f"Top retrieval score: {top_retrieval_score:.2f}.",
            f"Context size: {context_chars} chars (limit {context_limit}, truncated={context_truncated}).",
        ]

        if route_type == "general" and llm_settings.backend != "heuristic":
            generation_started = time.perf_counter()
            try:
                generation = self.llm_orchestrator.generate_general(
                    question=cleaned,
                    history=history_text,
                    settings=llm_settings,
                )
                answer = generation.text
                runtime = generation.runtime
                model = generation.model
                reasoning_chain.append("Generated conversational answer without retrieval grounding.")
            except LLMBackendError as exc:
                answer = self._compose_general_fallback(cleaned)
                runtime = "fallback-general"
                model = llm_settings.model
                reasoning_chain.append(f"General generation failed: {exc}. Used local fallback reply.")
            generation_ms = (time.perf_counter() - generation_started) * 1000.0

            return ChatResponse(
                answer=answer,
                citations=[],
                mode=mode,
                runtime=runtime,
                model=model,
                reasoning_chain=reasoning_chain,
                context_chars=context_chars,
                context_truncated=context_truncated,
                retrieval_chunks=0,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                route_type=route_type,
                route_confidence=route.confidence,
                route_reason=route.reason,
            )

        if llm_settings.backend == "heuristic":
            generation_started = time.perf_counter()
            if route_type == "general":
                answer = self._compose_general_fallback(cleaned)
            elif route_type == "hybrid" and not retrieved:
                answer = self._compose_hybrid_no_docs_fallback(cleaned)
            elif route_type == "hybrid" and top_retrieval_score < 1.1:
                answer = (
                    self._compose_heuristic_answer(cleaned, retrieved)
                    + "\n\nI can give a more precise answer if you mention a section, page, or specific metric."
                )
            else:
                answer = self._compose_heuristic_answer(cleaned, retrieved)

            if route_type == "hybrid" and retrieved:
                answer += "\n\nSimple explanation: this appears to be a technical core-analysis report with measured petrophysical data and interpretation notes."
            generation_ms = (time.perf_counter() - generation_started) * 1000.0
            reasoning_chain.append("Used heuristic grounded summarization (no external LLM call).")
            return ChatResponse(
                answer=answer,
                citations=citations,
                mode=mode,
                runtime="heuristic",
                model="none",
                reasoning_chain=reasoning_chain,
                context_chars=context_chars,
                context_truncated=context_truncated,
                retrieval_chunks=len(retrieved),
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                route_type=route_type,
                route_confidence=route.confidence,
                route_reason=route.reason,
            )

        generation_started = time.perf_counter()
        try:
            prompt_question = cleaned
            if route_type == "general":
                prompt_question = (
                    f"General conversation: {cleaned}\n"
                    "Respond naturally and helpfully."
                )
            elif route_type == "hybrid" and not retrieved:
                prompt_question = (
                    f"User asks: {cleaned}\n"
                    "No strong document evidence was found. Give a helpful general answer and ask if they want document-specific details."
                )

            generation = self.llm_orchestrator.generate(prompt_question, context, llm_settings)
            answer = generation.text
            runtime = generation.runtime
            model = generation.model
            reasoning_chain.append(f"Generated answer via runtime '{runtime}'.")
        except LLMBackendError as exc:
            if route_type == "general":
                fallback = self._compose_general_fallback(cleaned)
            elif route_type == "hybrid" and not retrieved:
                fallback = self._compose_hybrid_no_docs_fallback(cleaned)
            else:
                fallback = self._compose_heuristic_answer(cleaned, retrieved)
            answer = (
                "The selected local model runtime failed. "
                f"Details: {exc}\n\n"
                "Grounded fallback summary from extracted outputs:\n"
                f"{fallback}"
            )
            runtime = "fallback-heuristic"
            model = llm_settings.model
            reasoning_chain.append(f"LLM runtime error encountered: {exc}")
            reasoning_chain.append("Returned heuristic fallback answer grounded in extracted outputs.")
        generation_ms = (time.perf_counter() - generation_started) * 1000.0

        return ChatResponse(
            answer=answer,
            citations=citations,
            mode=mode,
            runtime=runtime,
            model=model,
            reasoning_chain=reasoning_chain,
            context_chars=context_chars,
            context_truncated=context_truncated,
            retrieval_chunks=len(retrieved),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            route_type=route_type,
            route_confidence=route.confidence,
            route_reason=route.reason,
        )

    def _handle_reporting_question(self, question: str, package: DocumentPackage | None) -> str | None:
        q = " ".join(question.lower().split())
        if "how many reports" in q and "database" in q:
            # runtime-level count not available here; caller should prefer global UI count.
            # provide useful answer from current context.
            if package is not None:
                return (
                    "I do not have global database count inside this answer context, "
                    "but I can confirm the selected report package is loaded and queryable. "
                    "Use the package counter shown in the app state panel for total loaded reports."
                )
            return (
                "I cannot see a loaded package context in this request. "
                "Load a folder first and I can report from the loaded set."
            )
        return None

    def _build_context(
        self,
        chunks: list[RetrievedChunk],
        max_chars: int = 24000,
        history_text: str = "",
    ) -> tuple[str, bool]:
        blocks: list[str] = []
        if history_text.strip():
            blocks.append(f"[RECENT CHAT]\n{history_text.strip()}")

        for idx, chunk in enumerate(chunks[:8], start=1):
            meta = f"source_file={chunk.source_file}; source_type={chunk.source_type}; score={chunk.score:.2f}"
            if chunk.section:
                meta += f"; section={chunk.section}"
            if chunk.page:
                meta += f"; page={chunk.page}"
            blocks.append(f"[SOURCE {idx}] {meta}\n{chunk.content.strip()}")

        merged = "\n\n".join(blocks).strip()
        if len(merged) <= max_chars:
            return merged, False
        return merged[:max_chars] + "\n\n[context truncated to fit prompt budget]", True

    def _compose_heuristic_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        highlights = self._extract_highlights(question, chunks)
        if not highlights:
            highlights = [self._truncate(chunk.content, 220) for chunk in chunks[:4]]

        lines = ["Grounded answer (extracted JSON/MD/TXT only):"]
        for item in highlights[:8]:
            lines.append(f"- {item}")
        return "\n".join(lines)

    def _compose_general_fallback(self, question: str) -> str:
        lowered = " ".join(question.strip().lower().split())
        if lowered in {"hi", "hello", "hey", "yo"}:
            return "Hey. How can I help you today?"
        if lowered in {"bye", "goodbye", "see you"}:
            return "Got it. Bye for now."
        if "how are you" in lowered or "how was your day" in lowered or "what's up" in lowered or "whats up" in lowered:
            return "I am doing well, thanks. I can chat normally or help with your document questions."

        return (
            "I can help with both general questions and document-grounded analysis. "
            "For document answers, ask about specific values, sections, tables, or report names."
        )

    def _compose_hybrid_no_docs_fallback(self, question: str) -> str:
        lowered = " ".join(question.strip().lower().split())
        if "porosity" in lowered:
            return (
                "Porosity is the fraction of a rock's volume that is pore space. "
                "It is usually reported as a percentage and indicates fluid storage capacity. "
                "If you load a report package, I can extract the exact porosity values from your documents."
            )
        if "permeability" in lowered:
            return (
                "Permeability describes how easily fluids flow through porous rock, commonly measured in mD. "
                "Higher permeability generally means easier fluid movement. "
                "Load a report package and I can return the specific measured values."
            )

        return (
            "I could not find strong supporting evidence in loaded documents for that exact query, "
            "so here is a general explanation: this appears to relate to core-analysis and petrophysical interpretation. "
            "If you want precise report-grounded values, ask with a specific metric or section."
        )

    def maybe_handle_ml_command(self, question: str, packages: list[DocumentPackage] | None = None) -> str | None:
        q = " ".join((question or "").lower().split())
        if "machine learning" in q or "predict" in q or "relative permeability" in q or "ml pipeline" in q:
            loaded = len(packages or [])
            return (
                "ML workflow command detected. You can use the ML Analytics tab to run:\n"
                "1) Build dataset from extracted JSON/MD/TXT\n"
                "2) Train predictive model\n"
                "3) Predict target values\n"
                "4) View analytics dashboard\n"
                f"Current loaded report packages: {loaded}."
            )
        return None

    def _extract_highlights(self, question: str, chunks: list[RetrievedChunk]) -> list[str]:
        query_tokens = {token for token in _tokenize(question) if len(token) > 2}
        highlights: list[str] = []

        for chunk in chunks[:8]:
            sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(chunk.content) if s.strip()]
            for sentence in sentences:
                sentence_tokens = set(_tokenize(sentence))
                if query_tokens and not sentence_tokens.intersection(query_tokens):
                    continue
                if _is_low_value_metadata_line(sentence):
                    continue
                highlights.append(self._truncate(sentence, 260))
                if len(highlights) >= 8:
                    return highlights

        return highlights

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        text = " ".join(text.split())
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _is_small_talk(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    if "how was your day" in normalized or "how's your day" in normalized or "hows your day" in normalized:
        return True
    if "how are you" in normalized:
        return True
    if len(normalized) <= 20 and normalized in {
        "hi",
        "hello",
        "hey",
        "yo",
        "hi there",
        "hello there",
        "good morning",
        "good afternoon",
        "good evening",
    }:
        return True
    if normalized in {"thanks", "thank you", "ok", "okay", "cool"}:
        return True
    return False


def _is_low_value_metadata_line(sentence: str) -> bool:
    cleaned = sentence.strip().lower()
    if not cleaned:
        return True
    if cleaned in {"---", "***", "___"}:
        return True
    prefixes = (
        "primary_language:",
        "is_rotation_valid:",
        "rotation_correction:",
        "is_table:",
        "is_diagram:",
    )
    return cleaned.startswith(prefixes)


def _build_history_context(messages: list[dict], limit_chars: int = 1200) -> str:
    if not messages:
        return ""

    lines: list[str] = []
    for message in messages[-8:]:
        role = str(message.get("role") or "assistant").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = " ".join(str(message.get("content") or "").split())
        if not content:
            continue
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {content}")

    merged = "\n".join(lines).strip()
    if len(merged) <= limit_chars:
        return merged
    return merged[-limit_chars:]
