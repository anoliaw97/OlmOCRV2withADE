from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget


class JsonViewer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.editor = QPlainTextEdit(self)
        self.editor.setReadOnly(True)
        font = QFont("Consolas", 10)
        self.editor.setFont(font)

        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)

    def set_json_text(self, text: str) -> None:
        if text.strip():
            self.editor.setPlainText(text)
        else:
            self.editor.setPlainText("No JSON content available for this package.")
