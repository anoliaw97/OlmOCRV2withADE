from __future__ import annotations

import re
from dataclasses import dataclass

from core.json_chunker import TextChunk, chunk_package_content
from core.loaders import DocumentPackage
from core.rag_index import IndexedSearchResult, LocalRagIndex


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(slots=True)
class RetrievedChunk:
    package_id: str
    source_file: str
    source_type: str
    content: str
    score: float
    section: str = ""
    page: str = ""
    table_name: str = ""


class RetrievalEngine:
    def __init__(self, rag_index: LocalRagIndex | None = None) -> None:
        self.rag_index = rag_index

    def retrieve_direct(self, package: DocumentPackage, question: str, top_k: int = 6) -> list[RetrievedChunk]:
        chunks = chunk_package_content(package)
        ranked = _rank_chunks(question, chunks)
        return ranked[:top_k]

    def retrieve_rag(
        self,
        question: str,
        top_k: int = 6,
        package_id: str | None = None,
    ) -> list[RetrievedChunk]:
        if self.rag_index is None:
            return []

        indexed_results = self.rag_index.search(question, limit=top_k, package_id=package_id)
        return [_from_index_result(item) for item in indexed_results]


def _from_index_result(item: IndexedSearchResult) -> RetrievedChunk:
    return RetrievedChunk(
        package_id=item.package_id,
        source_file=item.source_file,
        source_type=item.source_type,
        content=item.content,
        score=item.score,
        section=item.section,
        page=item.page,
        table_name=item.table_name,
    )


def _rank_chunks(question: str, chunks: list[TextChunk]) -> list[RetrievedChunk]:
    q_tokens = _tokenize(question)
    ranked: list[RetrievedChunk] = []

    for chunk in chunks:
        score = _score(q_tokens, question, chunk.content)
        if score <= 0:
            continue
        ranked.append(
            RetrievedChunk(
                package_id=chunk.package_id,
                source_file=chunk.source_file,
                source_type=chunk.source_type,
                content=chunk.content,
                score=score,
                section=chunk.metadata.get("section", ""),
                page=chunk.metadata.get("page", ""),
                table_name=chunk.metadata.get("table_name", ""),
            )
        )

    if not ranked:
        for chunk in chunks[:6]:
            ranked.append(
                RetrievedChunk(
                    package_id=chunk.package_id,
                    source_file=chunk.source_file,
                    source_type=chunk.source_type,
                    content=chunk.content,
                    score=0.1,
                    section=chunk.metadata.get("section", ""),
                    page=chunk.metadata.get("page", ""),
                    table_name=chunk.metadata.get("table_name", ""),
                )
            )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text) if len(match.group(0)) > 2}


def _score(question_tokens: set[str], question: str, content: str) -> float:
    if not content.strip():
        return 0.0

    content_tokens = _tokenize(content)
    if not content_tokens:
        return 0.0

    overlap = question_tokens.intersection(content_tokens)
    overlap_score = float(len(overlap) * 2)

    normalized_overlap = len(overlap) / (len(question_tokens) + 1)
    density = len(overlap) / (len(content_tokens) + 1)

    q_lower = question.lower().strip()
    phrase_bonus = 1.5 if q_lower and q_lower in content.lower() else 0.0
    number_bonus = 0.25 if any(ch.isdigit() for ch in question) and any(ch.isdigit() for ch in content) else 0.0

    return overlap_score + normalized_overlap + density + phrase_bonus + number_bonus
