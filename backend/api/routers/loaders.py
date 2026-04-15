from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.dependencies import get_runtime
from backend.schemas import LoaderPathRequest, LoaderResponse, PackagePreviewResponse, PackageRefRequest
from backend.utils import package_to_summary


router = APIRouter(prefix="/api/loaders", tags=["loaders"])


@router.post("/folder", response_model=LoaderResponse)
def load_folder(request: LoaderPathRequest) -> LoaderResponse:
    runtime = get_runtime()
    try:
        packages = runtime.load_folder(request.path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load folder: {exc}") from exc
    return LoaderResponse(count=len(packages), packages=[package_to_summary(p) for p in packages])


@router.post("/file", response_model=LoaderResponse)
def load_file(request: LoaderPathRequest) -> LoaderResponse:
    runtime = get_runtime()
    try:
        packages = runtime.load_primary_file(request.path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load file: {exc}") from exc
    return LoaderResponse(count=len(packages), packages=[package_to_summary(p) for p in packages])


@router.get("/packages", response_model=LoaderResponse)
def list_packages() -> LoaderResponse:
    runtime = get_runtime()
    packages = runtime.packages
    return LoaderResponse(count=len(packages), packages=[package_to_summary(p) for p in packages])


@router.post("/preview", response_model=PackagePreviewResponse)
def preview_package(request: PackageRefRequest) -> PackagePreviewResponse:
    runtime = get_runtime()
    package = runtime.get_package(request.package_id)
    if package is None:
        raise HTTPException(status_code=404, detail="Package not found. Load package first.")

    try:
        preview = runtime.preview_service.build_preview(package)
        tables = runtime.build_preview_tables(preview)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build preview: {exc}") from exc

    return PackagePreviewResponse(
        package_id=package.package_id,
        markdown_html=preview.markdown_html,
        json_text=preview.json_text,
        text_text=preview.text_text,
        pdf_path=str(preview.pdf_path) if preview.pdf_path else None,
        tables=tables,
    )
