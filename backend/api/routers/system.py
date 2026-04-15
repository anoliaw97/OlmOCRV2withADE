from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.dependencies import get_runtime
from backend.schemas import (
    BrowseDialogResponse,
    DirectoryBrowseResponse,
    DirectoryEntry,
    ModelOption,
    ModelOptionsResponse,
    DatasetPreviewResponse,
    RuntimeLogsResponse,
    RuntimeStateResponse,
    PopplerStatusResponse,
)
from core.model_registry import discover_llamacpp_models, list_ollama_models
from core.pdf_preview import resolve_pdftoppm_status


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


@router.post("/browse/file-dialog", response_model=BrowseDialogResponse)
def browse_file_dialog(path: str = Query(default=""), pattern: str = Query(default="*.csv")) -> BrowseDialogResponse:
    runtime = get_runtime()
    runtime.log("debug", f"File browse dialog requested. pattern={pattern}")
    selected = _select_file_dialog(path, pattern)
    if selected:
        runtime.log("status", f"Browse dialog selected file: {selected}")
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


@router.get("/dataset/preview", response_model=DatasetPreviewResponse)
def dataset_preview(path: str = Query(...), limit: int = Query(default=100, ge=1, le=500)) -> DatasetPreviewResponse:
    runtime = get_runtime()
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists() or not csv_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Only CSV preview is supported.")

    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:
        runtime.log("error", f"Dataset preview failed: {exc}")
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}") from exc

    preview = frame.head(limit).fillna("")
    rows = [
        {str(col): str(row[col]) for col in preview.columns}
        for _, row in preview.iterrows()
    ]
    runtime.log("status", f"Dataset preview loaded: {csv_path} rows={len(frame)}")
    return DatasetPreviewResponse(
        path=str(csv_path),
        rows=int(len(frame)),
        columns=[str(c) for c in frame.columns],
        preview_rows=rows,
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
            "Poppler pdftoppm executable not found. Install Poppler and set environment variable "
            "POPPLER_PATH or POPPLER_PDFTOPPM, then restart the app."
        ),
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


def _select_file_dialog(initial_path: str = "", pattern: str = "*.csv") -> str | None:
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
        else:
            initial_path_obj = Path(initial)
            if initial_path_obj.is_file():
                initial = str(initial_path_obj.parent)
        selected = filedialog.askopenfilename(
            initialdir=initial,
            filetypes=[("CSV files", pattern), ("All files", "*.*")],
        )
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
