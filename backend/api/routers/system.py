from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from backend.schemas import DirectoryBrowseResponse, DirectoryEntry, ModelOption, ModelOptionsResponse
from core.model_registry import discover_llamacpp_models, list_ollama_models


router = APIRouter(prefix="/api/system", tags=["system"])

MAX_BROWSE_ENTRIES = 500
DEFAULT_BROWSE_ROOT = Path(
    os.environ.get(
        "WORKFLOW_DEFAULT_BROWSE_ROOT",
        r"C:\Users\admin\Downloads\Fine Tunining Datasets\train",
    )
)


@router.get("/browse", response_model=DirectoryBrowseResponse)
def browse_directory(path: str | None = Query(default=None)) -> DirectoryBrowseResponse:
    target = Path(path).expanduser().resolve() if path else _resolve_default_browse_root()
    return _list_directory(target)


@router.get("/models/options", response_model=ModelOptionsResponse)
def model_options(
    backend: str = Query(default="ollama"),
    ollama_url: str = Query(default="http://127.0.0.1:11434/api/generate"),
    scan_path: str = Query(default=""),
) -> ModelOptionsResponse:
    backend_name = (backend or "ollama").strip().lower()
    if backend_name == "auto":
        backend_name = "ollama"

    if backend_name == "ollama":
        try:
            models = [ModelOption(**item) for item in list_ollama_models(ollama_url)]
            default_model = models[0].name if models else ""
            return ModelOptionsResponse(
                backend="ollama",
                connection_ok=True,
                message=f"Found {len(models)} Ollama model(s).",
                active=default_model,
                default_model=default_model,
                models=models,
            )
        except Exception as exc:
            return ModelOptionsResponse(
                backend="ollama",
                connection_ok=False,
                message=str(exc),
                active="",
                default_model="",
                models=[],
            )

    if backend_name == "llamacpp":
        models, resolved_scan = discover_llamacpp_models(scan_path)
        options = [ModelOption(**item) for item in models]
        default_model = options[0].path if options else ""
        return ModelOptionsResponse(
            backend="llamacpp",
            connection_ok=bool(options),
            message=f"Found {len(options)} GGUF model file(s).",
            active=default_model,
            default_model=default_model,
            scan_path=resolved_scan,
            models=options,
        )

    raise HTTPException(status_code=400, detail="backend must be ollama or llamacpp")


def _resolve_default_browse_root() -> Path:
    try:
        if DEFAULT_BROWSE_ROOT.exists() and DEFAULT_BROWSE_ROOT.is_dir():
            return DEFAULT_BROWSE_ROOT.resolve()
    except Exception:
        pass
    return Path.cwd().resolve()


def _list_directory(target: Path) -> DirectoryBrowseResponse:
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {target}")

    try:
        children = list(target.iterdir())
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {target}") from exc

    entries = sorted(children, key=lambda p: (not p.is_dir(), p.name.lower()))[:MAX_BROWSE_ENTRIES]
    default_root = _resolve_default_browse_root()

    return DirectoryBrowseResponse(
        current_path=str(target),
        parent_path=str(target.parent) if target.parent != target else None,
        default_root=str(default_root),
        entries=[DirectoryEntry(name=item.name, path=str(item), is_dir=item.is_dir()) for item in entries],
    )
