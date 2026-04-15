from __future__ import annotations

from pydantic import BaseModel, Field


class PackageSummary(BaseModel):
    package_id: str
    base_name: str
    folder: str
    pdf_path: str | None = None
    json_path: str | None = None
    markdown_path: str | None = None
    text_path: str | None = None
    related_files: list[str] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)


class LoaderPathRequest(BaseModel):
    path: str


class LoaderResponse(BaseModel):
    count: int
    packages: list[PackageSummary]


class PackageRefRequest(BaseModel):
    package_id: str


class PreviewTable(BaseModel):
    title: str
    source_type: str
    source_ref: str
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    raw_text: str = ""


class PackagePreviewResponse(BaseModel):
    package_id: str
    markdown_html: str
    json_text: str
    text_text: str
    pdf_path: str | None = None
    tables: list[PreviewTable] = Field(default_factory=list)


class BuildIndexResponse(BaseModel):
    indexed_chunks: int
    package_count: int


class LLMSettingsPayload(BaseModel):
    backend: str = "auto"
    model: str = "llama3.1:8b"
    system_prompt: str = ""
    max_tokens: int = 512
    temperature: float = 0.2
    ollama_url: str = "http://127.0.0.1:11434/api/generate"
    llama_cli_path: str = "llama-cli"


class ChatAskRequest(BaseModel):
    question: str
    mode: str = "direct"
    package_id: str | None = None
    llm_settings: LLMSettingsPayload = Field(default_factory=LLMSettingsPayload)


class CitationPayload(BaseModel):
    source_file: str
    source_type: str
    score: float
    section: str = ""
    page: str = ""


class ChatAskResponse(BaseModel):
    answer: str
    citations: list[CitationPayload] = Field(default_factory=list)
    mode: str
    runtime: str
    model: str = ""


class RetrievalQueryRequest(BaseModel):
    question: str
    mode: str = "rag"
    package_id: str | None = None
    top_k: int = 6


class RetrievalChunkPayload(BaseModel):
    package_id: str
    source_file: str
    source_type: str
    content: str
    score: float
    section: str = ""
    page: str = ""
    table_name: str = ""


class RetrievalQueryResponse(BaseModel):
    chunks: list[RetrievalChunkPayload] = Field(default_factory=list)


class ChatRecordPayload(BaseModel):
    timestamp: str
    mode: str
    runtime: str
    model: str
    question: str
    answer: str
    citations: str


class ExportChatRequest(BaseModel):
    destination: str
    records: list[ChatRecordPayload]


class ExportChatResponse(BaseModel):
    ok: bool
    message: str


class DirectoryEntry(BaseModel):
    name: str
    path: str
    is_dir: bool


class DirectoryBrowseResponse(BaseModel):
    current_path: str
    parent_path: str | None = None
    entries: list[DirectoryEntry] = Field(default_factory=list)
