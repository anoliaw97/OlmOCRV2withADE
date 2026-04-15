from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from core.chat_agent import Citation


class ChatWidget(QWidget):
    ask_requested = Signal(str, str)
    export_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.package_label = QLabel("No package selected")

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Direct selected-document query", "direct")
        self.mode_combo.addItem("Optional indexed RAG query", "rag")

        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(True)

        self.input_box = QPlainTextEdit()
        self.input_box.setPlaceholderText("Ask a question about extracted JSON/Markdown/TXT content...")
        self.input_box.setFixedHeight(90)

        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self._emit_ask)

        self.export_button = QPushButton("Export chat...")
        self.export_button.clicked.connect(lambda: self.export_requested.emit())

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(lambda: self.clear_requested.emit())

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Mode:"))
        top_row.addWidget(self.mode_combo)
        top_row.addStretch(1)

        action_row = QHBoxLayout()
        action_row.addWidget(self.ask_button)
        action_row.addWidget(self.export_button)
        action_row.addWidget(self.clear_button)
        action_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.package_label)
        layout.addLayout(top_row)
        layout.addWidget(self.chat_view, stretch=1)
        layout.addWidget(self.input_box)
        layout.addLayout(action_row)

        self.append_system_message("Chat ready. Answers are grounded in JSON/Markdown/TXT extracted outputs only.")

    def set_active_package_name(self, package_name: str) -> None:
        self.package_label.setText(f"Selected package: {package_name}")

    def append_user_message(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.chat_view.append(f"<p><b>You [{ts}]</b><br>{_escape_html(text)}</p>")
        self._scroll_to_bottom()

    def append_assistant_message(self, text: str, citations: list[Citation] | None = None) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        rendered = _escape_html(text).replace("\n", "<br>")
        self.chat_view.append(f"<p><b>Assistant [{ts}]</b><br>{rendered}</p>")

        if citations:
            citation_lines = [
                f"- {c.source_file} ({c.source_type}, score={c.score:.2f}{_meta_suffix(c)})"
                for c in citations
            ]
            self.chat_view.append(
                "<p><b>Sources</b><br>"
                + "<br>".join(_escape_html(line) for line in citation_lines)
                + "</p>"
            )

        self._scroll_to_bottom()

    def append_system_message(self, text: str) -> None:
        rendered = _escape_html(text).replace("\n", "<br>")
        self.chat_view.append(f"<p><i>{rendered}</i></p>")
        self._scroll_to_bottom()

    def clear_chat(self) -> None:
        self.chat_view.clear()

    def consume_input_text(self) -> tuple[str, str]:
        text = self.input_box.toPlainText().strip()
        mode = str(self.mode_combo.currentData())
        self.input_box.clear()
        return text, mode

    def _emit_ask(self) -> None:
        text, mode = self.consume_input_text()
        if not text:
            self.append_system_message("Please enter a question first.")
            return
        self.ask_requested.emit(text, mode)

    def _scroll_to_bottom(self) -> None:
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_view.setTextCursor(cursor)


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _meta_suffix(citation: Citation) -> str:
    parts: list[str] = []
    if citation.section:
        parts.append(f"section={citation.section}")
    if citation.page:
        parts.append(f"page={citation.page}")
    if not parts:
        return ""
    return ", " + ", ".join(parts)
