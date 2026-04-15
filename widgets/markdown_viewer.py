from __future__ import annotations

from PySide6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget


class MarkdownViewer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.browser = QTextBrowser(self)
        self.browser.setOpenExternalLinks(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.browser)

    def set_html(self, html: str) -> None:
        self.browser.setHtml(html)
