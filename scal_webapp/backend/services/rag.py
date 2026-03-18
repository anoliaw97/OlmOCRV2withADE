from __future__ import annotations


def synthesize_answer(question: str, hits: list[dict]) -> dict:
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

    answer = "\n".join([
        "Based on extracted SCAL table data:",
        *bullets,
        "",
        "Source traceability is included below.",
    ])

    return {"answer": answer, "sources": sources}
