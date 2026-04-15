from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.export_service import ChatRecord
from core.llm_backends import LLMSettings


class ApiClientError(RuntimeError):
    pass


@dataclass(slots=True)
class ApiCitation:
    source_file: str
    source_type: str
    score: float
    section: str = ""
    page: str = ""


@dataclass(slots=True)
class ApiChatResponse:
    answer: str
    citations: list[ApiCitation]
    mode: str
    runtime: str
    model: str


@dataclass(slots=True)
class ApiPreviewTable:
    title: str
    source_type: str
    source_ref: str
    headers: list[str]
    rows: list[list[str]]
    raw_text: str


@dataclass(slots=True)
class ApiPreview:
    package_id: str
    markdown_html: str
    json_text: str
    text_text: str
    pdf_path: Path | None
    tables: list[ApiPreviewTable]


@dataclass(slots=True)
class ApiPackage:
    package_id: str
    base_name: str
    folder: Path
    pdf_path: Path | None
    json_path: Path | None
    markdown_path: Path | None
    text_path: Path | None
    tokens: list[str]

    def display_label(self) -> str:
        labels = self.tokens or ["EMPTY"]
        return f"{self.base_name} [{', '.join(labels)}]"


class WorkflowApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def healthcheck(self) -> bool:
        try:
            self._request("GET", "/health")
            return True
        except Exception:
            return False

    def load_folder(self, folder: Path) -> list[ApiPackage]:
        payload = self._request("POST", "/api/loaders/folder", {"path": str(folder)})
        return _to_packages(payload.get("packages", []))

    def load_file(self, file_path: Path) -> list[ApiPackage]:
        payload = self._request("POST", "/api/loaders/file", {"path": str(file_path)})
        return _to_packages(payload.get("packages", []))

    def build_index(self) -> int:
        payload = self._request("POST", "/api/retrieval/index/build", {})
        return int(payload.get("indexed_chunks", 0))

    def preview(self, package_id: str) -> ApiPreview:
        payload = self._request("POST", "/api/loaders/preview", {"package_id": package_id})
        tables = [
            ApiPreviewTable(
                title=str(item.get("title", "")),
                source_type=str(item.get("source_type", "")),
                source_ref=str(item.get("source_ref", "")),
                headers=[str(h) for h in item.get("headers", [])],
                rows=[[str(cell) for cell in row] for row in item.get("rows", [])],
                raw_text=str(item.get("raw_text", "")),
            )
            for item in payload.get("tables", [])
        ]
        pdf_path = payload.get("pdf_path")
        return ApiPreview(
            package_id=str(payload.get("package_id", package_id)),
            markdown_html=str(payload.get("markdown_html", "")),
            json_text=str(payload.get("json_text", "")),
            text_text=str(payload.get("text_text", "")),
            pdf_path=Path(str(pdf_path)) if pdf_path else None,
            tables=tables,
        )

    def ask(self, question: str, mode: str, package_id: str | None, llm_settings: LLMSettings) -> ApiChatResponse:
        payload = self._request(
            "POST",
            "/api/chat/ask",
            {
                "question": question,
                "mode": mode,
                "package_id": package_id,
                "llm_settings": {
                    "backend": llm_settings.backend,
                    "model": llm_settings.model,
                    "system_prompt": llm_settings.system_prompt,
                    "max_tokens": llm_settings.max_tokens,
                    "temperature": llm_settings.temperature,
                    "ollama_url": llm_settings.ollama_url,
                    "llama_cli_path": llm_settings.llama_cli_path,
                },
            },
        )

        citations = [
            ApiCitation(
                source_file=str(c.get("source_file", "")),
                source_type=str(c.get("source_type", "")),
                score=float(c.get("score", 0.0)),
                section=str(c.get("section", "")),
                page=str(c.get("page", "")),
            )
            for c in payload.get("citations", [])
        ]
        return ApiChatResponse(
            answer=str(payload.get("answer", "")),
            citations=citations,
            mode=str(payload.get("mode", mode)),
            runtime=str(payload.get("runtime", "")),
            model=str(payload.get("model", "")),
        )

    def export_chat(self, records: list[ChatRecord], destination: Path) -> tuple[bool, str]:
        payload = self._request(
            "POST",
            "/api/export/chat",
            {
                "destination": str(destination),
                "records": [
                    {
                        "timestamp": r.timestamp,
                        "mode": r.mode,
                        "runtime": r.runtime,
                        "model": r.model,
                        "question": r.question,
                        "answer": r.answer,
                        "citations": r.citations,
                    }
                    for r in records
                ],
            },
        )
        return bool(payload.get("ok", False)), str(payload.get("message", ""))

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(url=url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=120) as response:
                raw = response.read().decode("utf-8", errors="ignore")
        except HTTPError as exc:
            detail = _extract_error_detail(exc)
            raise ApiClientError(detail) from exc
        except URLError as exc:
            raise ApiClientError(
                f"Backend connection failed for {url}. Start FastAPI server first. ({exc})"
            ) from exc

        if not raw.strip():
            return {}

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ApiClientError(f"Backend returned invalid JSON: {raw[:300]}") from exc

        if not isinstance(parsed, dict):
            raise ApiClientError("Unexpected backend response shape.")
        return parsed


def _extract_error_detail(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="ignore")
        payload = json.loads(body)
        detail = payload.get("detail") if isinstance(payload, dict) else ""
        if detail:
            return str(detail)
    except Exception:
        pass
    return f"Backend request failed with status {exc.code}."


def _to_packages(items: list[dict[str, Any]]) -> list[ApiPackage]:
    packages: list[ApiPackage] = []
    for item in items:
        packages.append(
            ApiPackage(
                package_id=str(item.get("package_id", "")),
                base_name=str(item.get("base_name", "")),
                folder=Path(str(item.get("folder", "."))),
                pdf_path=Path(str(item["pdf_path"])) if item.get("pdf_path") else None,
                json_path=Path(str(item["json_path"])) if item.get("json_path") else None,
                markdown_path=Path(str(item["markdown_path"])) if item.get("markdown_path") else None,
                text_path=Path(str(item["text_path"])) if item.get("text_path") else None,
                tokens=[str(t) for t in item.get("tokens", [])],
            )
        )
    return packages
