from __future__ import annotations

from .local_llm import get_rag_llm, get_usecase_llm


def _build_context(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        m = h["metadata"]
        lines.append(
            f"[{i}] file={m.get('file_name')} page={m.get('page_number')} table={m.get('table_id')} "
            f"type={m.get('extraction_type')}\n{h['text']}"
        )
    return "\n\n".join(lines)


def synthesize_answer(question: str, hits: list[dict], use_case_prompt: str | None = None) -> dict:
    if not hits:
        return {
            "answer": "No relevant extracted table data found for this query.",
            "sources": [],
        }

    bullets = []
    sources = []
    for h in hits:
        m = h["metadata"]
        snippet = h["text"]
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        bullets.append(f"- {snippet}")
        sources.append(
            {
                "file_name": m.get("file_name"),
                "page_number": m.get("page_number"),
                "table_id": m.get("table_id"),
                "extraction_type": m.get("extraction_type"),
            }
        )

    context = _build_context(hits)
    if use_case_prompt and use_case_prompt.strip():
        llm = get_usecase_llm()
        system = (
            "You are a SCAL analyst assistant. Answer using ONLY supplied extracted JSON/table evidence. "
            "If evidence is missing, say so clearly. Include citation markers like [1], [2]."
        )
        prompt = (
            f"Use-case instruction:\n{use_case_prompt}\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence chunks:\n{context}\n\n"
            "Provide concise answer with citations."
        )
        answer = llm.ask(prompt, system=system)
    else:
        llm = get_rag_llm()
        system = (
            "You are a SCAL RAG assistant. Use ONLY evidence chunks from extracted JSON/table data. "
            "Do not use external knowledge. Include citation markers like [1], [2]."
        )
        prompt = (
            f"Question:\n{question}\n\n"
            f"Evidence chunks:\n{context}\n\n"
            "Answer concisely and cite source chunk numbers."
        )
        answer = llm.ask(prompt, system=system)

    if not answer:
        answer = "\n".join([
            "Based on extracted SCAL table data:",
            *bullets,
            "",
            "Source traceability is included below.",
        ])

    return {"answer": answer, "sources": sources}
