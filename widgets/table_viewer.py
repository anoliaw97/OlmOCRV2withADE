from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.api_client import ApiPreviewTable
from core.table_renderer import ExtractedTable


class TableViewer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.tabs = QTabWidget(self)
        self.raw_fallback = QPlainTextEdit(self)
        self.raw_fallback.setReadOnly(True)
        self.raw_fallback.setPlaceholderText("Raw table fallback content will appear here.")

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs, stretch=3)
        layout.addWidget(self.raw_fallback, stretch=2)

        self.set_tables([])

    def set_tables(self, tables: list[ExtractedTable]) -> None:
        self.tabs.clear()
        if not tables:
            empty_widget = QLabel("No tables detected in JSON/Markdown for this package.")
            empty_widget.setWordWrap(True)
            self.tabs.addTab(empty_widget, "No tables")
            self.raw_fallback.setPlainText("")
            return

        raw_blocks: list[str] = []
        for index, table in enumerate(tables, start=1):
            title = f"{index}. {table.title}"

            if isinstance(table.dataframe, pd.DataFrame) and not table.dataframe.empty:
                widget = _dataframe_to_table_widget(table.dataframe)
            else:
                widget = QLabel("Table parsing fallback. Check raw view below.")
                widget.setWordWrap(True)

            self.tabs.addTab(widget, title)

            raw_piece = table.raw_text.strip()
            if raw_piece:
                raw_blocks.append(f"[{title}]\n{raw_piece}")

        self.raw_fallback.setPlainText("\n\n".join(raw_blocks))

    def set_api_tables(self, tables: list[ApiPreviewTable]) -> None:
        self.tabs.clear()
        if not tables:
            empty_widget = QLabel("No tables detected in JSON/Markdown for this package.")
            empty_widget.setWordWrap(True)
            self.tabs.addTab(empty_widget, "No tables")
            self.raw_fallback.setPlainText("")
            return

        raw_blocks: list[str] = []
        for index, table in enumerate(tables, start=1):
            title = f"{index}. {table.title}"

            if table.headers and table.rows:
                frame = pd.DataFrame(table.rows, columns=table.headers)
                widget = _dataframe_to_table_widget(frame)
            else:
                widget = QLabel("Table parsing fallback. Check raw view below.")
                widget.setWordWrap(True)

            self.tabs.addTab(widget, title)

            raw_piece = table.raw_text.strip()
            if raw_piece:
                raw_blocks.append(f"[{title}]\n{raw_piece}")

        self.raw_fallback.setPlainText("\n\n".join(raw_blocks))


def _dataframe_to_table_widget(frame: pd.DataFrame) -> QTableWidget:
    frame = frame.fillna("")
    widget = QTableWidget()
    widget.setColumnCount(len(frame.columns))
    widget.setHorizontalHeaderLabels([str(col) for col in frame.columns])
    widget.setRowCount(len(frame.index))

    for row_idx, (_, row) in enumerate(frame.iterrows()):
        for col_idx, value in enumerate(row.tolist()):
            widget.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

    widget.resizeColumnsToContents()
    widget.setAlternatingRowColors(True)
    return widget
