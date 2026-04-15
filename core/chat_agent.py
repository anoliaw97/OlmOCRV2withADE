from __future__ import annotations

import re
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
        if not cleaned:
            return ChatResponse(
                answer="Please enter a question.",
                citations=[],
                mode=mode,
                runtime="none",
                model="",
            )

        if mode == "rag":
            retrieved = self.retrieval_engine.retrieve_rag(cleaned)
            if not retrieved and package is not None:
                retrieved = self.retrieval_engine.retrieve_direct(package, cleaned)
        else:
            if package is None:
                return ChatResponse(
                    answer="No document package is selected. Load or select a package first.",
                    citations=[],
                    mode=mode,
                    runtime="none",
                    model="",
                )
            retrieved = self.retrieval_engine.retrieve_direct(package, cleaned)

        if not retrieved:
            return ChatResponse(
                answer=(
                    "I could not find matching evidence in extracted JSON/Markdown/TXT sources. "
                    "Try a more specific question or load a different package."
                ),
                citations=[],
                mode=mode,
                runtime="none",
                model=llm_settings.model,
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

        if llm_settings.backend == "heuristic":
            answer = self._compose_heuristic_answer(cleaned, retrieved)
            return ChatResponse(
                answer=answer,
                citations=citations,
                mode=mode,
                runtime="heuristic",
                model="none",
            )

        context = self._build_context(retrieved)
        try:
            generation = self.llm_orchestrator.generate(cleaned, context, llm_settings)
            answer = generation.text
            runtime = generation.runtime
            model = generation.model
        except LLMBackendError as exc:
            fallback = self._compose_heuristic_answer(cleaned, retrieved)
            answer = (
                f"LLM runtime error: {exc}\n\n"
                "Showing grounded fallback summary from extracted outputs:\n\n"
                f"{fallback}"
            )
            runtime = "fallback-heuristic"
            model = llm_settings.model

        return ChatResponse(
            answer=answer,
            citations=citations,
            mode=mode,
            runtime=runtime,
            model=model,
        )

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        blocks: list[str] = []
        for idx, chunk in enumerate(chunks[:8], start=1):
            meta = f"source_file={chunk.source_file}; source_type={chunk.source_type}; score={chunk.score:.2f}"
            if chunk.section:
                meta += f"; section={chunk.section}"
            if chunk.page:
                meta += f"; page={chunk.page}"
            blocks.append(f"[SOURCE {idx}] {meta}\n{chunk.content.strip()}")

        merged = "\n\n".join(blocks).strip()
        if len(merged) <= 24000:
            return merged
        return merged[:24000] + "\n\n[context truncated to fit prompt budget]"

    def _compose_heuristic_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        highlights = self._extract_highlights(question, chunks)
        if not highlights:
            highlights = [self._truncate(chunk.content, 220) for chunk in chunks[:4]]

        lines = ["Grounded answer (from extracted JSON/MD/TXT only):"]
        for item in highlights[:8]:
            lines.append(f"- {item}")

        lines.append("")
        lines.append("Notes:")
        lines.append("- PDF and images are preview-only in this workflow.")
        lines.append("- Validate final outputs using cited extracted sections.")
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
