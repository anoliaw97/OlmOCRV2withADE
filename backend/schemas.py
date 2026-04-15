from __future__ import annotations

from pydantic import BaseModel, Field


class PackageSummary(BaseModel):
    package_id: str
    base_name: str
    folder: str
    full_pdf_path: str | None = None
    pdf_path: str | None = None
    page_pdf_paths: list[str] = Field(default_factory=list)
    page_pdf_count: int = 0
    page_range: str = ""
    json_path: str | None = None
    markdown_path: str | None = None
    text_path: str | None = None
    json_paths: list[str] = Field(default_factory=list)
    markdown_paths: list[str] = Field(default_factory=list)
    text_paths: list[str] = Field(default_factory=list)
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
    full_pdf_path: str | None = None
    pdf_path: str | None = None
    page_pdf_paths: list[str] = Field(default_factory=list)
    tables: list[PreviewTable] = Field(default_factory=list)


class BuildIndexResponse(BaseModel):
    indexed_chunks: int
    package_count: int


class LLMSettingsPayload(BaseModel):
    backend: str = "ollama"
    model: str = "llama3.1:8b"
    system_prompt: str = ""
    max_tokens: int = 512
    temperature: float = 0.2
    ollama_url: str = "http://127.0.0.1:11434/api/generate"
    llama_cli_path: str = "llama-cli"
    context_limit: int = 24000


class ChatAskRequest(BaseModel):
    question: str
    mode: str = "direct"
    package_id: str | None = None
    session_id: str | None = None
    llm_settings: LLMSettingsPayload = Field(default_factory=LLMSettingsPayload)


class CitationPayload(BaseModel):
    source_file: str
    source_type: str
    score: float
    section: str = ""
    page: str = ""


class ChatMetricsPayload(BaseModel):
    context_limit: int = 0
    context_limit_source: str = "heuristic"
    context_chars: int = 0
    context_truncated: bool = False
    retrieval_chunks: int = 0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class ChatAskResponse(BaseModel):
    answer: str
    citations: list[CitationPayload] = Field(default_factory=list)
    mode: str
    runtime: str
    model: str = ""
    assistant_name: str = "Assistant"
    session_id: str | None = None
    reasoning_chain: list[str] = Field(default_factory=list)
    metrics: ChatMetricsPayload = Field(default_factory=ChatMetricsPayload)


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
    default_root: str | None = None
    entries: list[DirectoryEntry] = Field(default_factory=list)


class ModelOption(BaseModel):
    name: str
    label: str
    path: str = ""


class ModelOptionsResponse(BaseModel):
    backend: str
    connection_ok: bool = False
    message: str = ""
    active: str = ""
    default_model: str = ""
    scan_path: str = ""
    models: list[ModelOption] = Field(default_factory=list)


class SessionMessagePayload(BaseModel):
    role: str
    content: str
    time: str = ""
    runtime: str = ""
    model: str = ""
    citations: str = ""
    reasoning_chain: list[str] = Field(default_factory=list)


class ChatSessionPayload(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[SessionMessagePayload] = Field(default_factory=list)


class ChatSessionSummaryPayload(BaseModel):
    session_id: str
    title: str
    updated_at: str
    message_count: int


class ChatSessionsResponse(BaseModel):
    sessions: list[ChatSessionSummaryPayload] = Field(default_factory=list)


class ChatSessionCreateRequest(BaseModel):
    title: str = ""


class ChatSessionResponse(BaseModel):
    session: ChatSessionPayload


class GenericOkResponse(BaseModel):
    ok: bool
