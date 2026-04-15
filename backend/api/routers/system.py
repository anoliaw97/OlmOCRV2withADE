from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from backend.dependencies import get_runtime
from backend.schemas import (
    BrowseDialogResponse,
    DirectoryBrowseResponse,
    DirectoryEntry,
    ModelOption,
    ModelOptionsResponse,
    RuntimeLogsResponse,
    RuntimeStateResponse,
    PopplerConfigRequest,
    PopplerStatusResponse,
)
from core.model_registry import discover_llamacpp_models, list_ollama_models
from core.pdf_preview import resolve_pdftoppm_status, set_pdftoppm_path


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
    runtime = get_runtime()
    runtime.log("debug", f"Browsing directory: {target}")
    return _list_directory(target)


@router.post("/browse/dialog", response_model=BrowseDialogResponse)
def browse_dialog(path: str = Query(default="")) -> BrowseDialogResponse:
    runtime = get_runtime()
    runtime.log("debug", "Browse dialog requested.")
    selected = _select_folder_dialog(path)
    if selected:
        runtime.log("status", f"Browse dialog selected folder: {selected}")
    return BrowseDialogResponse(path=selected or "")


@router.get("/models/options", response_model=ModelOptionsResponse)
def model_options(
    backend: str = Query(default="ollama"),
    ollama_url: str = Query(default="http://127.0.0.1:11434/api/generate"),
    scan_path: str = Query(default=""),
) -> ModelOptionsResponse:
    runtime = get_runtime()
    backend_name = (backend or "ollama").strip().lower()
    if backend_name == "auto":
        backend_name = "ollama"

    if backend_name == "ollama":
        try:
            models = [ModelOption(**item) for item in list_ollama_models(ollama_url)]
            default_model = models[0].name if models else ""
            runtime.log("status", f"Ollama model scan completed: {len(models)} model(s) found.")
            return ModelOptionsResponse(
                backend="ollama",
                connection_ok=True,
                message=f"Found {len(models)} Ollama model(s).",
                active=default_model,
                default_model=default_model,
                models=models,
            )
        except Exception as exc:
            runtime.log("error", f"Ollama model scan failed: {exc}")
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
        runtime.log("status", f"llama.cpp model scan completed: {len(options)} GGUF file(s) found.")
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


@router.get("/state", response_model=RuntimeStateResponse)
def runtime_state() -> RuntimeStateResponse:
    runtime = get_runtime()
    current_package_id = runtime.packages[0].package_id if runtime.packages else ""
    return RuntimeStateResponse(
        packages_loaded=len(runtime.packages),
        current_package_id=current_package_id,
        rag_index_ready=runtime.rag_index.is_ready(),
        sessions=len(runtime.list_sessions()),
    )


@router.get("/logs", response_model=RuntimeLogsResponse)
def runtime_logs(kind: str = Query(default="status"), limit: int = Query(default=200)) -> RuntimeLogsResponse:
    runtime = get_runtime()
    items = runtime.list_logs(kind=kind, limit=limit)
    return RuntimeLogsResponse(kind=kind, items=items)


@router.post("/logs/clear")
def clear_logs(kind: str = Query(default="all")) -> dict[str, bool]:
    runtime = get_runtime()
    runtime.clear_logs(kind=kind)
    runtime.log("status", f"Logs cleared for kind='{kind}'.")
    return {"ok": True}


@router.get("/poppler/status", response_model=PopplerStatusResponse)
def poppler_status() -> PopplerStatusResponse:
    ok, configured, resolved = resolve_pdftoppm_status()
    if ok:
        return PopplerStatusResponse(
            ok=True,
            configured_path=configured,
            resolved_path=resolved,
            message="Poppler pdftoppm is available.",
        )
    return PopplerStatusResponse(
        ok=False,
        configured_path=configured,
        resolved_path=resolved,
        message=(
            "Poppler pdftoppm executable not found. Configure full path in settings, "
            "for example C:\\tools\\poppler\\Library\\bin\\pdftoppm.exe"
        ),
    )


@router.post("/poppler/config", response_model=PopplerStatusResponse)
def poppler_config(request: PopplerConfigRequest) -> PopplerStatusResponse:
    runtime = get_runtime()
    ok, payload = set_pdftoppm_path(request.pdftoppm_path)
    if not ok:
        runtime.log("error", f"Poppler config failed: {payload}")
        raise HTTPException(status_code=400, detail=payload)

    runtime.log("status", f"Poppler path configured: {payload}")
    status_ok, configured, resolved = resolve_pdftoppm_status()
    return PopplerStatusResponse(
        ok=status_ok,
        configured_path=configured,
        resolved_path=resolved,
        message="Poppler path configured successfully.",
    )


def _resolve_default_browse_root() -> Path:
    try:
        if DEFAULT_BROWSE_ROOT.exists() and DEFAULT_BROWSE_ROOT.is_dir():
            return DEFAULT_BROWSE_ROOT.resolve()
    except Exception:
        pass
    return Path.cwd().resolve()


def _select_folder_dialog(initial_path: str = "") -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
    except Exception:
        return None

    try:
        initial = initial_path.strip() if initial_path else ""
        if not initial:
            initial = str(_resolve_default_browse_root())
        selected = filedialog.askdirectory(initialdir=initial, mustexist=True)
        return selected or None
    finally:
        try:
            root.destroy()
        except Exception:
            pass


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
