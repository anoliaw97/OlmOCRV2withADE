from __future__ import annotations

import re
from dataclasses import dataclass

from core.loaders import DocumentPackage
from core.retriever import RetrievalEngine


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")

DOC_KEYWORDS = {
    "report",
    "document",
    "pdf",
    "table",
    "json",
    "markdown",
    "extract",
    "extracted",
    "analysis",
    "porosity",
    "permeability",
    "capillary",
    "core",
    "sample",
    "page",
    "citation",
    "values",
    "data",
    "database",
    "loaded",
    "count",
    "many",
    "number",
}

GENERAL_CHAT_KEYWORDS = {
    "hi",
    "hello",
    "hey",
    "bye",
    "thanks",
    "thank",
    "joke",
    "weather",
    "music",
    "movie",
    "today",
    "day",
    "about",
    "this",
    "that",
}

HYBRID_HINT_WORDS = {
    "explain",
    "summarize",
    "summary",
    "simple",
    "simply",
    "overall",
    "meaning",
    "interpret",
}


@dataclass(slots=True)
class RouteDecision:
    type: str
    confidence: float
    reason: str


class QueryRouter:
    def __init__(self, retrieval_engine: RetrievalEngine) -> None:
        self.retrieval_engine = retrieval_engine

    def route(
        self,
        question: str,
        package: DocumentPackage | None,
        preferred_mode: str,
        package_id: str | None = None,
    ) -> RouteDecision:
        text = " ".join(question.strip().lower().split())
        if not text:
            return RouteDecision(type="general", confidence=1.0, reason="empty-question")

        tokens = _tokenize(text)
        doc_hits = len(tokens.intersection(DOC_KEYWORDS))
        chat_hits = len(tokens.intersection(GENERAL_CHAT_KEYWORDS))
        hybrid_hits = len(tokens.intersection(HYBRID_HINT_WORDS))

        if _is_small_talk(text):
            return RouteDecision(type="general", confidence=0.98, reason="small-talk")

        if _is_weak_generic_query(text) and doc_hits == 0:
            return RouteDecision(type="hybrid", confidence=0.52, reason="weak-generic-query")

        if doc_hits > 0 and (hybrid_hits > 0 or chat_hits > 0):
            return RouteDecision(type="hybrid", confidence=0.86, reason="doc-plus-conversational-intent")

        if doc_hits >= 1:
            return RouteDecision(type="document", confidence=0.84, reason="document-keywords")

        probe = self._probe_relevance(
            question=question,
            package=package,
            preferred_mode=preferred_mode,
            package_id=package_id,
        )
        if probe >= 1.2:
            return RouteDecision(type="document", confidence=min(0.9, 0.62 + probe / 10.0), reason="retrieval-probe-strong")
        if probe >= 0.45:
            return RouteDecision(type="hybrid", confidence=min(0.78, 0.55 + probe / 10.0), reason="retrieval-probe-weak")

        return RouteDecision(type="general", confidence=0.68, reason="low-doc-relevance")

    def _probe_relevance(
        self,
        question: str,
        package: DocumentPackage | None,
        preferred_mode: str,
        package_id: str | None,
    ) -> float:
        chunks = []
        if preferred_mode == "rag":
            chunks = self.retrieval_engine.retrieve_rag(
                question=question,
                top_k=3,
                package_id=None,
                min_score=0.35,
                allow_fallback=False,
            )
            if not chunks and package is not None:
                chunks = self.retrieval_engine.retrieve_direct(
                    package=package,
                    question=question,
                    top_k=3,
                    min_score=0.35,
                    allow_fallback=False,
                )
        else:
            if package is not None:
                chunks = self.retrieval_engine.retrieve_direct(
                    package=package,
                    question=question,
                    top_k=3,
                    min_score=0.35,
                    allow_fallback=False,
                )
            if not chunks:
                chunks = self.retrieval_engine.retrieve_rag(
                    question=question,
                    top_k=3,
                    package_id=None,
                    min_score=0.35,
                    allow_fallback=False,
                )

        if not chunks:
            return 0.0
        return max(float(chunk.score) for chunk in chunks)


def _is_small_talk(text: str) -> bool:
    if any(phrase in text for phrase in ("how are you", "how was your day", "what's up", "whats up")):
        return True
    return text in {
        "hi",
        "hello",
        "hey",
        "bye",
        "goodbye",
        "thanks",
        "thank you",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
    }


def _is_weak_generic_query(text: str) -> bool:
    short = len(text.split()) <= 6
    if not short:
        return False
    patterns = (
        "what is this",
        "what is that",
        "what is this about",
        "what is that about",
        "explain this",
        "explain that",
    )
    return any(p in text for p in patterns)


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text) if len(match.group(0)) > 1}
