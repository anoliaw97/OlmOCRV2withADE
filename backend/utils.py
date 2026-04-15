from __future__ import annotations

from backend.schemas import PackageSummary


def package_to_summary(package) -> PackageSummary:
    tokens: list[str] = []
    if package.full_pdf_path:
        tokens.append("PDF")
    elif package.page_pdf_paths:
        tokens.append(f"PDFx{len(package.page_pdf_paths)}")

    if package.json_paths:
        tokens.append("JSON")
    if package.markdown_paths:
        tokens.append("MD")
    if package.text_paths:
        tokens.append("TXT")
    if not tokens:
        tokens.append("EMPTY")

    return PackageSummary(
        package_id=package.package_id,
        base_name=package.base_name,
        folder=str(package.folder),
        full_pdf_path=str(package.full_pdf_path) if package.full_pdf_path else None,
        pdf_path=str(package.pdf_path) if package.pdf_path else None,
        page_pdf_paths=[str(path) for path in package.page_pdf_paths],
        page_pdf_count=len(package.page_pdf_paths),
        page_range=package.page_range_text(),
        json_path=str(package.json_path) if package.json_path else None,
        markdown_path=str(package.markdown_path) if package.markdown_path else None,
        text_path=str(package.text_path) if package.text_path else None,
        json_paths=[str(path) for path in package.json_paths],
        markdown_paths=[str(path) for path in package.markdown_paths],
        text_paths=[str(path) for path in package.text_paths],
        related_files=[str(p) for p in package.related_files],
        tokens=tokens,
    )
