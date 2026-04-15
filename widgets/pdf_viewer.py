from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class PdfViewer(QWidget):
    """
    Practical first-version PDF preview:
    - Shows PDF location and metadata
    - Opens PDF in the system viewer

    This keeps dependencies lighter for phase 1 while still giving usable workflow behavior.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.current_pdf: Path | None = None

        self.info_label = QLabel("No PDF selected.")
        self.info_label.setWordWrap(True)

        self.open_button = QPushButton("Open PDF in default viewer")
        self.open_button.clicked.connect(self._open_external)
        self.open_button.setEnabled(False)

        self.note_label = QLabel(
            "Note: PDF is preview-only. Chat answers use extracted JSON/Markdown/TXT as source of truth."
        )
        self.note_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.info_label)
        layout.addWidget(self.open_button)
        layout.addWidget(self.note_label)
        layout.addStretch(1)

    def set_pdf(self, pdf_path: Path | None) -> None:
        self.current_pdf = pdf_path
        if pdf_path is None:
            self.info_label.setText("No PDF found in selected package.")
            self.open_button.setEnabled(False)
            return

        self.info_label.setText(f"PDF available:\n{pdf_path}")
        self.open_button.setEnabled(True)

    def _open_external(self) -> None:
        if self.current_pdf is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_pdf)))
