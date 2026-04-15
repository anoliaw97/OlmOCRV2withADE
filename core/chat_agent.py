from __future__ import annotations

import re
import time
from dataclasses import dataclass

from core.llm_backends import LLMBackendError, LLMOrchestrator, LLMSettings
from core.loaders import DocumentPackage
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


class ChatAgent:
    """Grounded chat over extracted JSON/Markdown/TXT with pluggable LLM backends."""

    def __init__(self, retrieval_engine: RetrievalEngine, llm_orchestrator: LLMOrchestrator | None = None) -> None:
        self.retrieval_engine = retrieval_engine
        self.llm_orchestrator = llm_orchestrator or LLMOrchestrator()

    def ask(
        self,
        question: str,
        package: DocumentPackage | None,
        mode: str,
        llm_settings: LLMSettings,
    ) -> ChatResponse:
        cleaned = question.strip()
        retrieval_ms = 0.0
        generation_ms = 0.0
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
            )

        if _is_small_talk(cleaned):
            return ChatResponse(
                answer=(
                    "Hi. I can help analyze your extracted JSON/Markdown/TXT data. "
                    "Load/select a package and ask a specific question (for example porosity, permeability, or capillary pressure values)."
                ),
                citations=[],
                mode=mode,
                runtime="assistant",
                model=llm_settings.model,
                reasoning_chain=["Detected small-talk greeting; responded conversationally without retrieval."],
                context_chars=0,
                context_truncated=False,
                retrieval_chunks=0,
                retrieval_ms=0.0,
                generation_ms=0.0,
            )

        retrieval_started = time.perf_counter()
        if mode == "rag":
            retrieved = self.retrieval_engine.retrieve_rag(cleaned)
            if not retrieved and package is not None:
                retrieved = self.retrieval_engine.retrieve_direct(package, cleaned)
        else:
            if package is None:
                retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
                return ChatResponse(
                    answer="No document package is selected. Load or select a package first.",
                    citations=[],
                    mode=mode,
                    runtime="none",
                    model="",
                    reasoning_chain=["Direct mode requires an active package."],
                    context_chars=0,
                    context_truncated=False,
                    retrieval_chunks=0,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                )
            retrieved = self.retrieval_engine.retrieve_direct(package, cleaned)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

        if not retrieved:
            return ChatResponse(
                answer=(
                    "I could not find relevant evidence in the loaded extracted JSON/Markdown/TXT files for that question. "
                    "Try a more specific query, choose another package, or rebuild the RAG index."
                ),
                citations=[],
                mode=mode,
                runtime="none",
                model=llm_settings.model,
                reasoning_chain=[
                    f"Mode: {mode}.",
                    "No matching extracted JSON/MD/TXT chunks were retrieved.",
                ],
                context_chars=0,
                context_truncated=False,
                retrieval_chunks=0,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
            )

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
        context, context_truncated = self._build_context(retrieved, max_chars=context_limit)
        context_chars = len(context)

        reasoning_chain = [
            f"Mode: {mode}.",
            f"Retrieved {len(retrieved)} chunk(s) from extracted JSON/MD/TXT.",
            f"Context size: {context_chars} chars (limit {context_limit}, truncated={context_truncated}).",
        ]

        if llm_settings.backend == "heuristic":
            generation_started = time.perf_counter()
            answer = self._compose_heuristic_answer(cleaned, retrieved)
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
            )

        generation_started = time.perf_counter()
        try:
            generation = self.llm_orchestrator.generate(cleaned, context, llm_settings)
            answer = generation.text
            runtime = generation.runtime
            model = generation.model
            reasoning_chain.append(f"Generated answer via runtime '{runtime}'.")
        except LLMBackendError as exc:
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
        )

    def _build_context(self, chunks: list[RetrievedChunk], max_chars: int = 24000) -> tuple[str, bool]:
        blocks: list[str] = []
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
