from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".json", ".md", ".markdown", ".txt"}


def _normalize_group_key(stem: str) -> str:
    normalized = stem.strip().lower()
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


@dataclass(slots=True)
class DocumentPackage:
    package_id: str
    base_name: str
    folder: Path
    pdf_path: Path | None = None
    json_path: Path | None = None
    markdown_path: Path | None = None
    text_path: Path | None = None
    related_files: list[Path] = field(default_factory=list)

    def extracted_paths(self) -> list[Path]:
        return [p for p in (self.json_path, self.markdown_path, self.text_path) if p is not None]

    def has_queryable_content(self) -> bool:
        return any(path is not None for path in (self.json_path, self.markdown_path, self.text_path))

    def display_label(self) -> str:
        tokens: list[str] = []
        if self.pdf_path:
            tokens.append("PDF")
        if self.json_path:
            tokens.append("JSON")
        if self.markdown_path:
            tokens.append("MD")
        if self.text_path:
            tokens.append("TXT")
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
        pdf_path = self._pick(files, {".pdf"})
        json_path = self._pick(files, {".json"})
        markdown_path = self._pick(files, {".md", ".markdown"})
        text_path = self._pick(files, {".txt"})

        if markdown_path is not None:
            base_name = markdown_path.stem
        elif json_path is not None:
            base_name = json_path.stem
        elif text_path is not None:
            base_name = text_path.stem
        elif pdf_path is not None:
            base_name = pdf_path.stem
        else:
            base_name = key

        package_id = f"{folder}:{key}"
        related = sorted(files, key=lambda p: p.name.lower())

        return DocumentPackage(
            package_id=package_id,
            base_name=base_name,
            folder=folder,
            pdf_path=pdf_path,
            json_path=json_path,
            markdown_path=markdown_path,
            text_path=text_path,
            related_files=related,
        )

    @staticmethod
    def _pick(files: list[Path], allowed_suffixes: set[str]) -> Path | None:
        matches = [p for p in files if p.suffix.lower() in allowed_suffixes]
        if not matches:
            return None
        matches.sort(key=lambda p: p.name.lower())
        return matches[0]
