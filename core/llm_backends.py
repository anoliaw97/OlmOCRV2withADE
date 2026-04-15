from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_SYSTEM_PROMPT = (
    "You are a technical extraction assistant. "
    "Answer only from the provided extracted context. "
    "If information is missing, explicitly say it is not found in extracted outputs."
)


@dataclass(slots=True)
class LLMSettings:
    backend: str = "auto"
    model: str = "llama3.1:8b"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_tokens: int = 512
    temperature: float = 0.2
    ollama_url: str = "http://127.0.0.1:11434/api/generate"
    llama_cli_path: str = "llama-cli"
    context_limit: int = 24000


@dataclass(slots=True)
class LLMGenerationResult:
    text: str
    runtime: str
    model: str


class LLMBackendError(RuntimeError):
    pass


class LLMOrchestrator:
    """Runs grounded answer generation via Ollama, llama.cpp, or transformers."""

    def __init__(self) -> None:
        self._transformers_runtime = _TransformersRuntime()

    def generate(self, question: str, context: str, settings: LLMSettings) -> LLMGenerationResult:
        backend = self._resolve_backend(settings)
        prompt = _build_grounded_prompt(
            question=question,
            context=context,
            max_context_chars=max(2000, int(settings.context_limit or 24000)),
        )

        if backend == "ollama":
            text = _run_ollama(prompt, settings)
            return LLMGenerationResult(text=text, runtime="ollama", model=settings.model)

        if backend == "llamacpp":
            text = _run_llama_cpp(prompt, settings)
            return LLMGenerationResult(text=text, runtime="llama.cpp", model=settings.model)

        if backend == "transformers":
            text = self._transformers_runtime.generate(prompt, settings)
            return LLMGenerationResult(text=text, runtime="transformers", model=settings.model)

        raise LLMBackendError(f"Unsupported backend: {backend}")

    def _resolve_backend(self, settings: LLMSettings) -> str:
        explicit = settings.backend.strip().lower()
        if explicit and explicit != "auto":
            return explicit

        model = settings.model.strip()
        if model.lower().endswith(".gguf"):
            return "llamacpp"

        if model and _is_url_reachable(_ollama_tags_url(settings.ollama_url), timeout=1.5):
            return "ollama"

        if model:
            model_path = Path(model)
            if model_path.exists() and model_path.is_dir():
                return "transformers"

        if model and not model.lower().endswith(".gguf"):
            if _is_url_reachable(_ollama_tags_url(settings.ollama_url), timeout=1.5):
                return "ollama"

        raise LLMBackendError(
            "Auto backend could not resolve an available runtime. "
            "Start Ollama, provide a GGUF model path + llama-cli, or choose transformers backend explicitly."
        )


def _build_grounded_prompt(question: str, context: str, max_context_chars: int) -> str:
    clipped_context = context[:max_context_chars]
    return (
        "Use ONLY the extracted context below (JSON/MD/TXT derived).\n"
        "Do NOT use external assumptions.\n"
        "If a requested value is absent, answer: 'Not found in extracted outputs.'\n\n"
        "Return concise but complete technical output.\n"
        "Include table-like bullet formatting for key numeric values when possible.\n\n"
        "[EXTRACTED CONTEXT START]\n"
        f"{clipped_context}\n"
        "[EXTRACTED CONTEXT END]\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _run_ollama(prompt: str, settings: LLMSettings) -> str:
    model = settings.model.strip()
    if not model:
        raise LLMBackendError("Ollama backend requires a model name (for example: llama3.1:8b).")

    payload = {
        "model": model,
        "prompt": prompt,
        "system": settings.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": float(settings.temperature),
            "num_predict": int(settings.max_tokens),
        },
    }

    data = json.dumps(payload).encode("utf-8")
    request = Request(
        settings.ollama_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8", errors="ignore")
    except URLError as exc:
        raise LLMBackendError(
            f"Failed to call Ollama at {settings.ollama_url}. Ensure `ollama serve` is running. ({exc})"
        ) from exc

    try:
        parsed: dict[str, Any] = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMBackendError(f"Ollama returned non-JSON response: {body[:400]}") from exc

    if parsed.get("error"):
        raise LLMBackendError(str(parsed["error"]))

    text = str(parsed.get("response", "")).strip()
    if not text:
        raise LLMBackendError("Ollama returned an empty answer.")
    return text


def _run_llama_cpp(prompt: str, settings: LLMSettings) -> str:
    model = settings.model.strip()
    if not model:
        raise LLMBackendError("llama.cpp backend requires a GGUF model path.")

    model_path = Path(model)
    if not model_path.exists():
        raise LLMBackendError(f"GGUF model path does not exist: {model_path}")

    cli_path = settings.llama_cli_path.strip() or "llama-cli"
    resolved_cli = shutil.which(cli_path) if Path(cli_path).name == cli_path else cli_path
    if not resolved_cli:
        raise LLMBackendError(
            "Could not find `llama-cli`. Provide full path in settings or add it to PATH."
        )

    command = [
        resolved_cli,
        "-m",
        str(model_path),
        "-p",
        prompt,
        "-n",
        str(int(settings.max_tokens)),
        "--temp",
        str(float(settings.temperature)),
        "--no-display-prompt",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=240,
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMBackendError("llama.cpp timed out while generating response.") from exc
    except OSError as exc:
        raise LLMBackendError(f"Failed to execute llama.cpp command: {exc}") from exc

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "Unknown llama.cpp failure").strip()
        raise LLMBackendError(err)

    text = (result.stdout or "").strip()
    if not text:
        raise LLMBackendError("llama.cpp returned an empty answer.")
    return text


class _TransformersRuntime:
    def __init__(self) -> None:
        self._loaded_model_id: str | None = None
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def generate(self, prompt: str, settings: LLMSettings) -> str:
        model_id = settings.model.strip()
        if not model_id:
            raise LLMBackendError(
                "Transformers backend requires local model folder path or model id."
            )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise LLMBackendError(
                "Transformers backend dependencies missing. Install: pip install transformers torch accelerate safetensors sentencepiece"
            ) from exc

        if self._loaded_model_id != model_id:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(model_id)
            except Exception as exc:
                raise LLMBackendError(f"Failed to load transformers model '{model_id}': {exc}") from exc

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(self._device)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._loaded_model_id = model_id

        assert self._tokenizer is not None
        assert self._model is not None

        tokenizer = self._tokenizer
        model = self._model

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        do_sample = settings.temperature > 0.0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(settings.max_tokens),
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(max(settings.temperature, 0.01))

        try:
            import torch

            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_kwargs)
        except Exception as exc:
            raise LLMBackendError(f"Transformers generation failed: {exc}") from exc

        prompt_len = int(inputs["input_ids"].shape[1])
        new_tokens = output_ids[0][prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not text:
            raise LLMBackendError("Transformers model returned an empty answer.")
        return text


def _ollama_tags_url(generate_url: str) -> str:
    parsed = urlparse(generate_url)
    if not parsed.scheme or not parsed.netloc:
        return "http://127.0.0.1:11434/api/tags"
    return f"{parsed.scheme}://{parsed.netloc}/api/tags"


def _is_url_reachable(url: str, timeout: float = 1.5) -> bool:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 500
    except Exception:
        return False
