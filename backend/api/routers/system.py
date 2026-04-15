from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from backend.schemas import DirectoryBrowseResponse, DirectoryEntry


router = APIRouter(prefix="/api/system", tags=["system"])

MAX_BROWSE_ENTRIES = 500


@router.get("/browse", response_model=DirectoryBrowseResponse)
def browse_directory(path: str | None = Query(default=None)) -> DirectoryBrowseResponse:
    if not path:
        default = Path.cwd().resolve()
        return _list_directory(default)

    target = Path(path).expanduser().resolve()
    return _list_directory(target)


def _list_directory(target: Path) -> DirectoryBrowseResponse:
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {target}")

    try:
        children = list(target.iterdir())
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {target}") from exc

    entries = sorted(children, key=lambda p: (not p.is_dir(), p.name.lower()))[:MAX_BROWSE_ENTRIES]

    return DirectoryBrowseResponse(
        current_path=str(target),
        parent_path=str(target.parent) if target.parent != target else None,
        entries=[DirectoryEntry(name=item.name, path=str(item), is_dir=item.is_dir()) for item in entries],
    )
