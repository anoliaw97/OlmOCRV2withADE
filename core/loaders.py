from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".json", ".md", ".markdown", ".txt"}
PAGE_SUFFIX_PATTERN = re.compile(r"^(?P<base>.+?)[_-]page(?P<page>\d+)$", re.IGNORECASE)


def _normalize_group_key(stem: str) -> str:
    base_stem, _ = _parse_page_suffix(stem)
    normalized = base_stem.strip().lower()
    suffixes = (
        "_extracted",
        "-extracted",
        "_output",
        "-output",
        "_results",
        "-results",
        "_report",
        "-report",
    )
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized or stem.lower()


def _parse_page_suffix(stem: str) -> tuple[str, int | None]:
    raw = stem.strip()
    match = PAGE_SUFFIX_PATTERN.match(raw)
    if not match:
        return raw, None
    try:
        return match.group("base"), int(match.group("page"))
    except Exception:
        return match.group("base"), None


@dataclass(slots=True)
class DocumentPackage:
    package_id: str
    base_name: str
    folder: Path
    full_pdf_path: Path | None = None
    pdf_path: Path | None = None
    json_path: Path | None = None
    markdown_path: Path | None = None
    text_path: Path | None = None
    page_pdf_paths: list[Path] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)
    json_paths: list[Path] = field(default_factory=list)
    markdown_paths: list[Path] = field(default_factory=list)
    text_paths: list[Path] = field(default_factory=list)
    related_files: list[Path] = field(default_factory=list)

    def extracted_paths(self) -> list[Path]:
        ordered = [*self.json_paths, *self.markdown_paths, *self.text_paths]
        if ordered:
            return ordered
        return [p for p in (self.json_path, self.markdown_path, self.text_path) if p is not None]

    def has_queryable_content(self) -> bool:
        return bool(self.extracted_paths())

    def page_range_text(self) -> str:
        if not self.page_numbers:
            return ""
        first = min(self.page_numbers)
        last = max(self.page_numbers)
        if first == last:
            return str(first)
        return f"{first}-{last}"

    def display_label(self) -> str:
        tokens: list[str] = []
        if self.full_pdf_path:
            tokens.append("PDF")
        elif self.page_pdf_paths:
            tokens.append(f"PDFx{len(self.page_pdf_paths)}")

        if self.json_paths:
            tokens.append("JSON" if len(self.json_paths) == 1 else f"JSONx{len(self.json_paths)}")
        if self.markdown_paths:
            tokens.append("MD" if len(self.markdown_paths) == 1 else f"MDx{len(self.markdown_paths)}")
        if self.text_paths:
            tokens.append("TXT" if len(self.text_paths) == 1 else f"TXTx{len(self.text_paths)}")
        if not tokens:
            tokens.append("EMPTY")
        return f"{self.base_name} [{', '.join(tokens)}]"


class PackageLoader:
    """Loads document packages from a folder or a primary file path."""

    def load_from_folder(self, folder: Path) -> list[DocumentPackage]:
        folder = folder.expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

        grouped: dict[str, list[Path]] = {}
        for path in folder.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            key = _normalize_group_key(path.stem)
            grouped.setdefault(key, []).append(path)

        packages: list[DocumentPackage] = []
        for key, files in grouped.items():
            package = self._assemble_package(key, folder, files)
            packages.append(package)

        packages.sort(key=lambda p: p.base_name.lower())
        return packages

    def load_from_primary_file(self, primary_file: Path) -> DocumentPackage:
        primary_file = primary_file.expanduser().resolve()
        if not primary_file.exists() or not primary_file.is_file():
            raise FileNotFoundError(f"File not found: {primary_file}")

        if primary_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {primary_file.suffix}")

        folder = primary_file.parent
        key = _normalize_group_key(primary_file.stem)
        siblings = [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and _normalize_group_key(p.stem) == key
        ]

        if not siblings:
            siblings = [primary_file]

        return self._assemble_package(key, folder, siblings)

    def _assemble_package(self, key: str, folder: Path, files: list[Path]) -> DocumentPackage:
        ordered = sorted(files, key=_file_sort_key)

        all_pdfs = [p for p in ordered if p.suffix.lower() == ".pdf"]
        full_pdf_path = next((p for p in all_pdfs if _parse_page_suffix(p.stem)[1] is None), None)
        page_pdf_paths = [p for p in all_pdfs if _parse_page_suffix(p.stem)[1] is not None]
        page_numbers = sorted(
            {
                page
                for page in (_parse_page_suffix(path.stem)[1] for path in page_pdf_paths)
                if page is not None
            }
        )

        json_paths = [p for p in ordered if p.suffix.lower() == ".json"]
        markdown_paths = [p for p in ordered if p.suffix.lower() in {".md", ".markdown"}]
        text_paths = [p for p in ordered if p.suffix.lower() == ".txt"]

        json_path = _prefer_non_page(json_paths)
        markdown_path = _prefer_non_page(markdown_paths)
        text_path = _prefer_non_page(text_paths)

        pdf_path = full_pdf_path or (page_pdf_paths[0] if page_pdf_paths else None)

        if markdown_path is not None:
            base_name = _parse_page_suffix(markdown_path.stem)[0]
        elif json_path is not None:
            base_name = _parse_page_suffix(json_path.stem)[0]
        elif text_path is not None:
            base_name = _parse_page_suffix(text_path.stem)[0]
        elif full_pdf_path is not None:
            base_name = _parse_page_suffix(full_pdf_path.stem)[0]
        elif page_pdf_paths:
            base_name = _parse_page_suffix(page_pdf_paths[0].stem)[0]
        else:
            base_name = key

        package_id = f"{folder}:{key}"
        related = sorted(files, key=_file_sort_key)

        return DocumentPackage(
            package_id=package_id,
            base_name=base_name,
            folder=folder,
            full_pdf_path=full_pdf_path,
            pdf_path=pdf_path,
            json_path=json_path,
            markdown_path=markdown_path,
            text_path=text_path,
            page_pdf_paths=page_pdf_paths,
            page_numbers=page_numbers,
            json_paths=json_paths,
            markdown_paths=markdown_paths,
            text_paths=text_paths,
            related_files=related,
        )

    @staticmethod
    def _pick(files: list[Path], allowed_suffixes: set[str]) -> Path | None:
        matches = [p for p in files if p.suffix.lower() in allowed_suffixes]
        if not matches:
            return None
        matches.sort(key=lambda p: p.name.lower())
        return matches[0]


def _prefer_non_page(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    non_page = [path for path in paths if _parse_page_suffix(path.stem)[1] is None]
    if non_page:
        return sorted(non_page, key=_file_sort_key)[0]
    return sorted(paths, key=_file_sort_key)[0]


def _file_sort_key(path: Path) -> tuple[int, int, str]:
    _, page = _parse_page_suffix(path.stem)
    if page is None:
        return (0, 0, path.name.lower())
    return (1, page, path.name.lower())
