from __future__ import annotations

from pathlib import Path

from backend.schemas import PreviewTable
from core.chat_agent import ChatAgent
from core.chat_export_agent import ChatExportAgent
from core.chat_sessions import ChatSessionStore
from core.export_service import ChatRecord, ExportService
from core.llm_backends import LLMSettings
from core.loaders import DocumentPackage, PackageLoader
from core.ml_pipeline_service import MlPipelineService
from core.preview_service import PackagePreview, PreviewService
from core.query_router import RouteDecision
from core.rag_index import LocalRagIndex
from core.runtime_logs import RuntimeLogs
from core.retriever import RetrievalEngine


MAX_TABLE_HEADERS = 30
MAX_TABLE_ROWS = 200
MAX_CELL_TEXT = 1000


class WorkflowRuntime:
    def __init__(self) -> None:
        self.loader = PackageLoader()
        self.preview_service = PreviewService()
        self.rag_index = LocalRagIndex(Path("data/index/rag_index.sqlite"))
        self.retrieval_engine = RetrievalEngine(self.rag_index)
        self.chat_agent = ChatAgent(self.retrieval_engine)
        self.export_service = ExportService()
        self.chat_export_agent = ChatExportAgent(self.retrieval_engine, self.export_service)
        self.ml_service = MlPipelineService(Path("."))
        self.session_store = ChatSessionStore(Path("data/chat_sessions.json"))
        self.logs = RuntimeLogs()

        self.packages: list[DocumentPackage] = []
        self._packages_by_id: dict[str, DocumentPackage] = {}
        self.log("status", "Workflow runtime initialized.")

    def set_packages(self, packages: list[DocumentPackage]) -> list[DocumentPackage]:
        self.packages = packages
        self._packages_by_id = {pkg.package_id: pkg for pkg in packages}
        self.log("status", f"Package state updated: {len(packages)} package(s) loaded.")
        return self.packages

    def load_folder(self, folder_path: str) -> list[DocumentPackage]:
        loaded = self.loader.load_from_folder(Path(folder_path))
        return self.set_packages(loaded)

    def load_primary_file(self, file_path: str) -> list[DocumentPackage]:
        package = self.loader.load_from_primary_file(Path(file_path))
        return self.set_packages([package])

    def get_package(self, package_id: str | None) -> DocumentPackage | None:
        if not package_id:
            return None
        return self._packages_by_id.get(package_id)

    def build_index(self) -> int:
        if not self.packages:
            return 0
        return self.rag_index.build_or_update(self.packages)

    def ask(
        self,
        question: str,
        mode: str,
        package_id: str | None,
        settings: LLMSettings,
        session_history: list[dict] | None = None,
        route_decision: RouteDecision | None = None,
    ):
        package = self.get_package(package_id)
        return self.chat_agent.ask(
            question=question,
            package=package,
            mode=mode,
            llm_settings=settings,
            session_history=session_history or [],
            route_decision=route_decision,
            package_id=package_id,
        )

    def retrieve(self, question: str, mode: str, package_id: str | None, top_k: int) -> list:
        if mode == "direct":
            package = self.get_package(package_id)
            if package is None:
                return []
            return self.retrieval_engine.retrieve_direct(package=package, question=question, top_k=top_k)

        return self.retrieval_engine.retrieve_rag(question=question, top_k=top_k, package_id=package_id)

    def build_preview_tables(self, preview: PackagePreview) -> list[PreviewTable]:
        tables: list[PreviewTable] = []

        for table in preview.tables:
            headers: list[str] = []
            rows: list[list[str]] = []

            if table.dataframe is not None and not table.dataframe.empty:
                frame = table.dataframe.fillna("")
                headers = [str(col)[:MAX_CELL_TEXT] for col in frame.columns.tolist()[:MAX_TABLE_HEADERS]]
                for row_values in frame.iloc[:MAX_TABLE_ROWS, :MAX_TABLE_HEADERS].itertuples(index=False):
                    rows.append([str(value)[:MAX_CELL_TEXT] for value in row_values])

            raw = (table.raw_text or "")[:MAX_CELL_TEXT * 8]
            tables.append(
                PreviewTable(
                    title=table.title,
                    source_type=table.source_type,
                    source_ref=table.source_ref,
                    headers=headers,
                    rows=rows,
                    raw_text=raw,
                )
            )

        return tables

    def export_records(self, destination: str, records: list[ChatRecord]) -> tuple[bool, str]:
        return self.export_service.export_chat_records(records, Path(destination))

    def list_sessions(self) -> list[dict]:
        return self.session_store.list_sessions()

    def create_session(self, title: str = "") -> dict:
        return self.session_store.create_session(title)

    def get_session(self, session_id: str) -> dict | None:
        return self.session_store.get_session(session_id)

    def append_session_messages(self, session_id: str, messages: list[dict]) -> dict:
        return self.session_store.append_messages(session_id, messages)

    def update_session_title(self, session_id: str, title: str) -> dict | None:
        return self.session_store.update_title(session_id, title)

    def delete_session(self, session_id: str) -> bool:
        return self.session_store.delete_session(session_id)

    def recent_session_messages(self, session_id: str, limit: int = 8) -> list[dict]:
        return self.session_store.recent_messages(session_id, limit=limit)

    def close(self) -> None:
        self.log("status", "Workflow runtime shutdown requested.")
        self.rag_index.close()

    def log(self, kind: str, message: str) -> None:
        self.logs.add(kind, message)

    def list_logs(self, kind: str, limit: int = 200) -> list[dict[str, str]]:
        return [
            {"time": item.time, "kind": item.kind, "message": item.message}
            for item in self.logs.list(kind, limit)
        ]

    def clear_logs(self, kind: str = "all") -> None:
        self.logs.clear(kind)
