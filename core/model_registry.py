from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_LLAMACPP_SCAN_ROOTS = [
    Path(r"C:\Users\admin\Downloads\Fine Tunining Datasets\train"),
    Path(r"C:\models"),
    Path(r"D:\models"),
]

MAX_DISCOVERED_MODELS = 300
MAX_SCAN_DEPTH = 5


def ollama_tags_url(generate_url: str) -> str:
    parsed = urlparse(generate_url)
    if not parsed.scheme or not parsed.netloc:
        return "http://127.0.0.1:11434/api/tags"
    return f"{parsed.scheme}://{parsed.netloc}/api/tags"


def ollama_show_url(generate_url: str) -> str:
    parsed = urlparse(generate_url)
    if not parsed.scheme or not parsed.netloc:
        return "http://127.0.0.1:11434/api/show"
    return f"{parsed.scheme}://{parsed.netloc}/api/show"


def list_ollama_models(generate_url: str) -> list[dict[str, str]]:
    tags_url = ollama_tags_url(generate_url)
    request = Request(tags_url, method="GET")

    try:
        with urlopen(request, timeout=10) as response:
            raw = response.read().decode("utf-8", errors="ignore")
    except URLError as exc:
        raise RuntimeError(f"Ollama not reachable at {tags_url}: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON for tags: {raw[:240]}") from exc

    models: list[dict[str, str]] = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        models.append({"name": name, "label": name, "path": ""})

    models.sort(key=lambda m: m["name"].lower())
    return models


def discover_llamacpp_models(scan_path: str = "") -> tuple[list[dict[str, str]], str]:
    roots = _resolve_scan_roots(scan_path)
    collected: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    for root in roots:
        if len(collected) >= MAX_DISCOVERED_MODELS:
            break
        if not root.exists() or not root.is_dir():
            continue
        _scan_gguf(root, depth=0, out=collected, seen=seen_paths)

    collected.sort(key=lambda m: m["label"].lower())
    selected_scan_path = str(roots[0]) if roots else ""
    return collected[:MAX_DISCOVERED_MODELS], selected_scan_path


def detect_model_context_limit(backend: str, model_name: str, ollama_url: str) -> tuple[int, str]:
    backend_name = (backend or "").strip().lower()
    model = (model_name or "").strip()
    guessed = _guess_context_limit_from_name(model)
    source = "name-heuristic"

    if backend_name == "ollama" and model:
        try:
            show_request = Request(
                ollama_show_url(ollama_url),
                data=json.dumps({"name": model}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(show_request, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8", errors="ignore"))
            candidates = _collect_context_candidates(payload)
            if candidates:
                return max(candidates), "ollama-show"
        except Exception:
            pass

    if guessed > 0:
        return guessed, source

    if backend_name == "llamacpp":
        return 8192, "llamacpp-default"
    if backend_name == "transformers":
        return 4096, "transformers-default"
    if backend_name == "heuristic":
        return 0, "heuristic"
    return 8192, "fallback-default"


def _resolve_scan_roots(scan_path: str) -> list[Path]:
    roots: list[Path] = []
    manual = scan_path.strip()
    if manual:
        roots.append(Path(manual).expanduser())

    env_root = os.environ.get("SCAL_LLAMACPP_MODEL_DIR", "").strip()
    if env_root:
        roots.append(Path(env_root).expanduser())

    roots.extend(DEFAULT_LLAMACPP_SCAN_ROOTS)

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolved)
    return deduped


def _scan_gguf(root: Path, depth: int, out: list[dict[str, str]], seen: set[str]) -> None:
    if depth > MAX_SCAN_DEPTH or len(out) >= MAX_DISCOVERED_MODELS:
        return

    try:
        children = list(root.iterdir())
    except Exception:
        return

    for child in children:
        if len(out) >= MAX_DISCOVERED_MODELS:
            return
        if child.is_dir():
            _scan_gguf(child, depth + 1, out, seen)
            continue
        if child.suffix.lower() != ".gguf":
            continue

        norm = str(child).lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append({"name": child.name, "label": child.name, "path": str(child)})


def _collect_context_candidates(value: Any) -> list[int]:
    candidates: list[int] = []
    if isinstance(value, dict):
        for k, v in value.items():
            key = str(k).lower()
            if any(token in key for token in ("context", "ctx", "num_ctx", "n_ctx", "max_input")):
                parsed = _to_int(v)
                if parsed > 0:
                    candidates.append(parsed)
            candidates.extend(_collect_context_candidates(v))
        return candidates

    if isinstance(value, list):
        for item in value:
            candidates.extend(_collect_context_candidates(item))
        return candidates

    if isinstance(value, str):
        for match in re.finditer(
            r"(?i)(?:context|ctx|window|max[_ -]?input|num[_ -]?ctx|n[_ -]?ctx)[^0-9]{0,20}(\d{3,7})",
            value,
        ):
            parsed = _to_int(match.group(1))
            if parsed > 0:
                candidates.append(parsed)
    return candidates


def _guess_context_limit_from_name(model_name: str) -> int:
    model = (model_name or "").lower()
    if not model:
        return 0

    for match in re.finditer(r"(\d{1,3})\s*k", model):
        val = _to_int(match.group(1))
        if val > 0:
            return val * 1024

    explicit = re.search(r"(?:ctx|context|window)[^0-9]{0,10}(\d{3,6})", model)
    if explicit:
        parsed = _to_int(explicit.group(1))
        if parsed > 0:
            return parsed
    return 0


def _to_int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return 0
