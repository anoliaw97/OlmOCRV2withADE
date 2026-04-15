from __future__ import annotations

from backend.schemas import PackageSummary


def package_to_summary(package) -> PackageSummary:
    tokens: list[str] = []
    if package.pdf_path:
        tokens.append("PDF")
    if package.json_path:
        tokens.append("JSON")
    if package.markdown_path:
        tokens.append("MD")
    if package.text_path:
        tokens.append("TXT")
    if not tokens:
        tokens.append("EMPTY")

    return PackageSummary(
        package_id=package.package_id,
        base_name=package.base_name,
        folder=str(package.folder),
        pdf_path=str(package.pdf_path) if package.pdf_path else None,
        json_path=str(package.json_path) if package.json_path else None,
        markdown_path=str(package.markdown_path) if package.markdown_path else None,
        text_path=str(package.text_path) if package.text_path else None,
        related_files=[str(p) for p in package.related_files],
        tokens=tokens,
    )
