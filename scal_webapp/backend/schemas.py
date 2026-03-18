from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExtractionSettings(BaseModel):
    use_case: str = "capillary_pressure"
    mode: str = "layman"
    page_range: str | None = None
    extraction_types: list[str] = Field(default_factory=lambda: ["capillary_pressure"])
    prompt_profile: str = "default"
    model_name: str = "offline_heuristic"
    normalize: bool = True
    build_index: bool = True


class TableJSON(BaseModel):
    file_name: str
    page_number: int
    table_id: str
    extraction_type: str
    table_title: str | None
    columns: list[str]
    rows: list[dict[str, Any]]
    units: dict[str, str] | None = None
    metadata: dict[str, Any]


class QueryRequest(BaseModel):
    question: str
    report_name: str | None = None
    file_name: str | None = None
    page_number: int | None = None
    extraction_type: str | None = None
    sample_id: str | None = None
    top_k: int = 6
