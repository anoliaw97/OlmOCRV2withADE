from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from core.chat_agent import Citation
from core.llm_backends import DEFAULT_SYSTEM_PROMPT, LLMSettings


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

        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Auto (Ollama/GGUF/Tensor)", "auto")
        self.backend_combo.addItem("Ollama", "ollama")
        self.backend_combo.addItem("llama.cpp (GGUF)", "llamacpp")
        self.backend_combo.addItem("Transformers (tensor/safetensors)", "transformers")
        self.backend_combo.addItem("Heuristic (no LLM)", "heuristic")
        self.backend_combo.currentIndexChanged.connect(self._update_backend_hint)

        self.model_input = QLineEdit("llama3.1:8b")
        self.model_input.setPlaceholderText("Model name or local model path")

        self.ollama_url_input = QLineEdit("http://127.0.0.1:11434/api/generate")
        self.llama_cli_input = QLineEdit("llama-cli")

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(64, 8192)
        self.max_tokens_spin.setValue(512)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(0.2)

        self.system_prompt_input = QPlainTextEdit()
        self.system_prompt_input.setPlainText(DEFAULT_SYSTEM_PROMPT)
        self.system_prompt_input.setFixedHeight(68)

        self.backend_hint_label = QLabel("")
        self.backend_hint_label.setWordWrap(True)

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
        top_row.addWidget(QLabel("LLM Backend:"))
        top_row.addWidget(self.backend_combo)
        top_row.addStretch(1)

        llm_form = QFormLayout()
        llm_form.addRow("Model (name or path)", self.model_input)
        llm_form.addRow("Ollama URL", self.ollama_url_input)
        llm_form.addRow("llama-cli path", self.llama_cli_input)
        llm_form.addRow("Max new tokens", self.max_tokens_spin)
        llm_form.addRow("Temperature", self.temperature_spin)
        llm_form.addRow("System prompt", self.system_prompt_input)

        action_row = QHBoxLayout()
        action_row.addWidget(self.ask_button)
        action_row.addWidget(self.export_button)
        action_row.addWidget(self.clear_button)
        action_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.package_label)
        layout.addLayout(top_row)
        layout.addLayout(llm_form)
        layout.addWidget(self.backend_hint_label)
        layout.addWidget(self.chat_view, stretch=1)
        layout.addWidget(self.input_box)
        layout.addLayout(action_row)

        self._update_backend_hint()
        self.append_system_message(
            "Chat ready. Answers are grounded in JSON/Markdown/TXT extracted outputs only. "
            "Configure LLM backend to use local models."
        )

    def set_active_package_name(self, package_name: str) -> None:
        self.package_label.setText(f"Selected package: {package_name}")

    def get_llm_settings(self) -> LLMSettings:
        return LLMSettings(
            backend=str(self.backend_combo.currentData()),
            model=self.model_input.text().strip(),
            system_prompt=self.system_prompt_input.toPlainText().strip() or DEFAULT_SYSTEM_PROMPT,
            max_tokens=self.max_tokens_spin.value(),
            temperature=float(self.temperature_spin.value()),
            ollama_url=self.ollama_url_input.text().strip() or "http://127.0.0.1:11434/api/generate",
            llama_cli_path=self.llama_cli_input.text().strip() or "llama-cli",
        )

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

    def _update_backend_hint(self) -> None:
        backend = str(self.backend_combo.currentData())

        if backend == "ollama":
            self.model_input.setPlaceholderText("Ollama model name, e.g. llama3.1:8b")
            self.backend_hint_label.setText(
                "Ollama mode: start `ollama serve`, then use pulled model name (for example `ollama run llama3.1:8b`)."
            )
            return

        if backend == "llamacpp":
            self.model_input.setPlaceholderText(r"C:\models\your-model.gguf")
            self.backend_hint_label.setText(
                "llama.cpp mode: set GGUF model path and llama-cli path (or keep `llama-cli` if it is on PATH)."
            )
            return

        if backend == "transformers":
            self.model_input.setPlaceholderText(r"C:\models\local-transformers-model")
            self.backend_hint_label.setText(
                "Transformers mode: model should be a local folder or model id with compatible tensor/safetensors files."
            )
            return

        if backend == "heuristic":
            self.model_input.setPlaceholderText("No model required")
            self.backend_hint_label.setText("Heuristic mode: no external LLM call; grounded extraction summary only.")
            return

        self.model_input.setPlaceholderText("Auto chooses runtime by model/path and local availability")
        self.backend_hint_label.setText(
            "Auto mode: tries Ollama when available, GGUF through llama.cpp for .gguf paths, or transformers for local model folders."
        )

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
