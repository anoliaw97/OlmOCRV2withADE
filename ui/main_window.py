from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.chat_agent import ChatAgent
from core.export_service import ChatRecord, ExportService
from core.loaders import DocumentPackage, PackageLoader
from core.preview_service import PreviewService
from core.rag_index import LocalRagIndex
from core.retriever import RetrievalEngine
from ui import dialogs
from widgets.chat_widget import ChatWidget
from widgets.json_viewer import JsonViewer
from widgets.markdown_viewer import MarkdownViewer
from widgets.pdf_viewer import PdfViewer
from widgets.table_viewer import TableViewer


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Python Workflow Query Desktop")
        self.resize(1500, 920)

        self.loader = PackageLoader()
        self.preview_service = PreviewService()
        self.rag_index = LocalRagIndex(Path("data/index/rag_index.sqlite"))
        self.retrieval_engine = RetrievalEngine(self.rag_index)
        self.chat_agent = ChatAgent(self.retrieval_engine)
        self.export_service = ExportService()

        self.packages: list[DocumentPackage] = []
        self.current_package: DocumentPackage | None = None
        self.chat_records: list[ChatRecord] = []

        self._build_ui()
        self._set_status("Ready. Load a folder or primary file to begin.")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.rag_index.close()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1140])

        self.setCentralWidget(splitter)

    def _build_left_panel(self) -> QWidget:
        wrapper = QWidget(self)
        layout = QVBoxLayout(wrapper)

        top_row = QHBoxLayout()
        self.load_folder_btn = QPushButton("Load Folder")
        self.load_folder_btn.clicked.connect(self._on_load_folder)
        self.load_file_btn = QPushButton("Load File")
        self.load_file_btn.clicked.connect(self._on_load_file)
        top_row.addWidget(self.load_folder_btn)
        top_row.addWidget(self.load_file_btn)

        self.build_index_btn = QPushButton("Build/Update Optional RAG Index")
        self.build_index_btn.clicked.connect(self._on_build_index)

        self.package_list = QListWidget()
        self.package_list.currentRowChanged.connect(self._on_package_row_changed)

        self.package_meta = QLabel("Package metadata will appear here.")
        self.package_meta.setWordWrap(True)

        layout.addLayout(top_row)
        layout.addWidget(self.build_index_btn)
        layout.addWidget(QLabel("Detected Document Packages"))
        layout.addWidget(self.package_list, stretch=1)
        layout.addWidget(self.package_meta)

        return wrapper

    def _build_right_panel(self) -> QWidget:
        wrapper = QWidget(self)
        layout = QVBoxLayout(wrapper)

        self.tabs = QTabWidget(self)

        self.pdf_viewer = PdfViewer(self)
        self.markdown_viewer = MarkdownViewer(self)
        self.table_viewer = TableViewer(self)
        self.json_viewer = JsonViewer(self)
        self.chat_widget = ChatWidget(self)

        self.chat_widget.ask_requested.connect(self._on_chat_question)
        self.chat_widget.export_requested.connect(self._on_export_chat)
        self.chat_widget.clear_requested.connect(self._on_clear_chat)

        self.tabs.addTab(self.pdf_viewer, "PDF Preview")
        self.tabs.addTab(self.markdown_viewer, "Markdown Preview")
        self.tabs.addTab(self.table_viewer, "Rendered Tables")
        self.tabs.addTab(self.json_viewer, "Raw JSON")
        self.tabs.addTab(self.chat_widget, "Chat")

        layout.addWidget(self.tabs)
        return wrapper

    def _on_load_folder(self) -> None:
        folder = dialogs.select_folder(self)
        if folder is None:
            return

        try:
            packages = self.loader.load_from_folder(folder)
        except Exception as exc:
            dialogs.show_error(self, "Load folder failed", str(exc))
            return

        if not packages:
            dialogs.show_info(self, "No packages", "No supported files found in selected folder.")
            return

        self.packages = packages
        self._populate_package_list()
        self.package_list.setCurrentRow(0)
        self._set_status(f"Loaded {len(packages)} package(s) from {folder}")

    def _on_load_file(self) -> None:
        file_path = dialogs.select_primary_file(self)
        if file_path is None:
            return

        try:
            package = self.loader.load_from_primary_file(file_path)
        except Exception as exc:
            dialogs.show_error(self, "Load file failed", str(exc))
            return

        self.packages = [package]
        self._populate_package_list()
        self.package_list.setCurrentRow(0)
        self._set_status(f"Loaded package from {file_path}")

    def _populate_package_list(self) -> None:
        self.package_list.clear()
        for package in self.packages:
            item = QListWidgetItem(package.display_label())
            self.package_list.addItem(item)

    def _on_package_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.packages):
            self.current_package = None
            return

        package = self.packages[row]
        self.current_package = package
        self.chat_widget.set_active_package_name(package.base_name)

        try:
            preview = self.preview_service.build_preview(package)
        except Exception as exc:
            dialogs.show_error(self, "Preview error", str(exc))
            return

        self.pdf_viewer.set_pdf(preview.pdf_path)
        self.markdown_viewer.set_html(preview.markdown_html)
        self.table_viewer.set_tables(preview.tables)
        self.json_viewer.set_json_text(preview.json_text)

        meta_lines = [
            f"Folder: {package.folder}",
            f"JSON: {package.json_path if package.json_path else 'N/A'}",
            f"Markdown: {package.markdown_path if package.markdown_path else 'N/A'}",
            f"TXT: {package.text_path if package.text_path else 'N/A'}",
            f"PDF: {package.pdf_path if package.pdf_path else 'N/A'}",
        ]
        self.package_meta.setText("\n".join(meta_lines))
        self._set_status(f"Selected package: {package.base_name}")

    def _on_build_index(self) -> None:
        if not self.packages:
            dialogs.show_info(self, "No packages loaded", "Load one or more packages first.")
            return

        chunk_count = self.rag_index.build_or_update(self.packages)
        self._set_status(f"Indexed {chunk_count} chunk(s) from {len(self.packages)} package(s).")
        self.chat_widget.append_system_message(
            f"Optional RAG index updated. Indexed chunks: {chunk_count}."
        )

    def _on_chat_question(self, question: str, mode: str) -> None:
        if mode == "direct" and self.current_package is None:
            self.chat_widget.append_system_message("Direct mode requires selecting a document package first.")
            return

        if mode == "rag" and not self.rag_index.is_ready() and self.current_package is None:
            self.chat_widget.append_system_message(
                "RAG mode has no index and no package selected. Load/select a package or build index first."
            )
            return

        self.chat_widget.append_user_message(question)
        llm_settings = self.chat_widget.get_llm_settings()

        try:
            response = self.chat_agent.ask(
                question,
                package=self.current_package,
                mode=mode,
                llm_settings=llm_settings,
            )
        except Exception as exc:
            self.chat_widget.append_system_message(f"Chat error: {exc}")
            return

        self.chat_widget.append_assistant_message(response.answer, response.citations)
        self.chat_widget.append_system_message(
            f"Runtime used: {response.runtime} | Model: {response.model or 'not set'}"
        )

        citation_text = "; ".join(
            f"{c.source_file}:{c.source_type}:{c.score:.2f}" for c in response.citations
        )
        self.chat_records.append(
            ChatRecord(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                mode=response.mode,
                runtime=response.runtime,
                model=response.model,
                question=question,
                answer=response.answer,
                citations=citation_text,
            )
        )
        self._set_status(
            f"Answered question in {response.mode} mode via {response.runtime} ({response.model or 'no model'})."
        )

    def _on_export_chat(self) -> None:
        if not self.chat_records:
            dialogs.show_info(self, "No chat history", "Ask at least one question before exporting.")
            return

        destination = dialogs.select_export_file(self)
        if destination is None:
            return

        ok, message = self.export_service.export_chat_records(self.chat_records, destination)
        if ok:
            dialogs.show_info(self, "Export complete", message)
            self._set_status(message)
        else:
            dialogs.show_error(self, "Export failed", message)

    def _on_clear_chat(self) -> None:
        button = QMessageBox.question(
            self,
            "Clear chat",
            "Clear chat history in this session?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if button != QMessageBox.StandardButton.Yes:
            return

        self.chat_records.clear()
        self.chat_widget.clear_chat()
        self.chat_widget.append_system_message("Chat cleared.")
        self._set_status("Chat history cleared.")

    def _set_status(self, text: str) -> None:
        self.statusBar().showMessage(text, 10000)
