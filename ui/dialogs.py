from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget


def select_folder(parent: QWidget) -> Path | None:
    folder = QFileDialog.getExistingDirectory(parent, "Select folder with extracted outputs")
    return Path(folder) if folder else None


def select_primary_file(parent: QWidget) -> Path | None:
    filters = "Supported files (*.json *.md *.markdown *.txt *.pdf);;All files (*.*)"
    path, _ = QFileDialog.getOpenFileName(parent, "Select primary package file", "", filters)
    return Path(path) if path else None


def select_export_file(parent: QWidget) -> Path | None:
    filters = "CSV (*.csv);;Excel (*.xlsx);;Word Document (*.docx)"
    path, _ = QFileDialog.getSaveFileName(parent, "Export chat output", "chat_export.csv", filters)
    return Path(path) if path else None


def show_info(parent: QWidget, title: str, message: str) -> None:
    QMessageBox.information(parent, title, message)


def show_error(parent: QWidget, title: str, message: str) -> None:
    QMessageBox.critical(parent, title, message)
