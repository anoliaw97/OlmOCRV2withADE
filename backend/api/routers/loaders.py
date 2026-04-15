from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from backend.dependencies import get_runtime
from backend.schemas import LoaderPathRequest, LoaderResponse, PackagePreviewResponse, PackageRefRequest
from backend.utils import package_to_summary
from core.pdf_preview import PdfPreviewError, render_pdf_page_png


router = APIRouter(prefix="/api/loaders", tags=["loaders"])


@router.post("/folder", response_model=LoaderResponse)
def load_folder(request: LoaderPathRequest) -> LoaderResponse:
    runtime = get_runtime()
    runtime.log("status", f"Load folder requested: {request.path}")
    try:
        packages = runtime.load_folder(request.path)
    except Exception as exc:
        runtime.log("error", f"Load folder failed: {exc}")
        raise HTTPException(status_code=400, detail=f"Failed to load folder: {exc}") from exc
    runtime.log("status", f"Loaded folder with {len(packages)} package(s).")
    return LoaderResponse(count=len(packages), packages=[package_to_summary(p) for p in packages])


@router.post("/file", response_model=LoaderResponse)
def load_file(request: LoaderPathRequest) -> LoaderResponse:
    runtime = get_runtime()
    runtime.log("status", f"Load file requested: {request.path}")
    try:
        packages = runtime.load_primary_file(request.path)
    except Exception as exc:
        runtime.log("error", f"Load file failed: {exc}")
        raise HTTPException(status_code=400, detail=f"Failed to load file: {exc}") from exc
    runtime.log("status", f"Loaded file into {len(packages)} package(s).")
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
        runtime.log("error", f"Preview failed; package not found: {request.package_id}")
        raise HTTPException(status_code=404, detail="Package not found. Load package first.")

    try:
        preview = runtime.preview_service.build_preview(package)
        tables = runtime.build_preview_tables(preview)
    except Exception as exc:
        runtime.log("error", f"Preview build failed for {request.package_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to build preview: {exc}") from exc

    runtime.log("debug", f"Preview built for package {request.package_id} with {len(tables)} table(s).")

    return PackagePreviewResponse(
        package_id=package.package_id,
        markdown_text=preview.markdown_text,
        markdown_html=preview.markdown_html,
        json_text=preview.json_text,
        text_text=preview.text_text,
        full_pdf_path=str(package.full_pdf_path) if package.full_pdf_path else None,
        pdf_path=str(preview.pdf_path) if preview.pdf_path else None,
        page_pdf_paths=[str(path) for path in package.page_pdf_paths],
        tables=tables,
    )


@router.get("/preview/pdf-image")
def preview_pdf_image(
    package_id: str = Query(...),
    page: int = Query(default=1),
    dpi: int = Query(default=140),
) -> Response:
    runtime = get_runtime()
    package = runtime.get_package(package_id)
    if package is None:
        raise HTTPException(status_code=404, detail="Package not found. Load package first.")

    try:
        pdf_path, resolved_page = _resolve_pdf_target(package, page)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        image_bytes = render_pdf_page_png(pdf_path=pdf_path, page=resolved_page, dpi=dpi)
    except PdfPreviewError as exc:
        runtime.log("error", f"PDF preview failed for {pdf_path}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    runtime.log("debug", f"Rendered PDF preview image for package={package_id} page={page} dpi={dpi}")
    return Response(content=image_bytes, media_type="image/png")


def _resolve_pdf_target(package, requested_page: int) -> tuple[Path, int]:
    page_num = max(1, int(requested_page))
    if package.full_pdf_path and package.full_pdf_path.exists():
        return package.full_pdf_path, page_num

    if package.page_pdf_paths:
        if package.page_numbers and len(package.page_numbers) == len(package.page_pdf_paths):
            by_page = {int(pg): path for pg, path in zip(package.page_numbers, package.page_pdf_paths)}
            if page_num in by_page:
                return by_page[page_num], 1

        index = min(max(page_num - 1, 0), len(package.page_pdf_paths) - 1)
        return package.page_pdf_paths[index], 1

    if package.pdf_path and package.pdf_path.exists():
        return package.pdf_path, page_num

    raise ValueError("No PDF is available for the selected package.")
