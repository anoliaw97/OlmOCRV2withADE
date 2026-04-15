from __future__ import annotations

import re
from dataclasses import dataclass

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


class ChatAgent:
    """Simple grounded chat over extracted JSON/Markdown/TXT content."""

    def __init__(self, retrieval_engine: RetrievalEngine) -> None:
        self.retrieval_engine = retrieval_engine

    def ask(
        self,
        question: str,
        package: DocumentPackage | None,
        mode: str,
    ) -> ChatResponse:
        cleaned = question.strip()
        if not cleaned:
            return ChatResponse(answer="Please enter a question.", citations=[], mode=mode)

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
            )

        answer = self._compose_answer(cleaned, retrieved)
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
        return ChatResponse(answer=answer, citations=citations, mode=mode)

    def _compose_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        highlights = self._extract_highlights(question, chunks)
        if not highlights:
            highlights = [self._truncate(chunk.content, 220) for chunk in chunks[:4]]

        lines = ["Grounded answer (from extracted JSON/MD/TXT only):"]
        for item in highlights[:8]:
            lines.append(f"- {item}")

        lines.append("")
        lines.append("Notes:")
        lines.append("- PDF and images are treated as preview assets only.")
        lines.append("- Final validation should use the cited extracted sections below.")
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
