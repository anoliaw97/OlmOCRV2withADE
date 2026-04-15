from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_PDFTOPPM_CANDIDATES = [
    r"C:\Program Files\poppler\Library\bin\pdftoppm.exe",
    r"C:\Program Files (x86)\poppler\Library\bin\pdftoppm.exe",
    r"C:\poppler\Library\bin\pdftoppm.exe",
]


class PdfPreviewError(RuntimeError):
    pass


def render_pdf_page_png(pdf_path: Path, page: int = 1, dpi: int = 140) -> bytes:
    pdf = pdf_path.expanduser().resolve()
    if not pdf.exists() or not pdf.is_file():
        raise PdfPreviewError(f"PDF not found: {pdf}")

    pdftoppm = _resolve_pdftoppm()
    page_num = max(1, int(page))
    dpi_num = max(72, min(int(dpi), 320))

    with tempfile.TemporaryDirectory(prefix="wf_pdf_preview_") as tmpdir:
        output_prefix = Path(tmpdir) / "page"
        command = [
            str(pdftoppm),
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            "-singlefile",
            "-r",
            str(dpi_num),
            "-png",
            str(pdf),
            str(output_prefix),
        ]

        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise PdfPreviewError("Poppler preview timed out while rendering PDF page.") from exc
        except OSError as exc:
            raise PdfPreviewError(f"Failed to execute pdftoppm: {exc}") from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "pdftoppm failed").strip()
            raise PdfPreviewError(detail)

        png_path = output_prefix.with_suffix(".png")
        if not png_path.exists():
            legacy_path = Path(f"{output_prefix}-1.png")
            if legacy_path.exists():
                png_path = legacy_path

        if not png_path.exists():
            raise PdfPreviewError("Poppler did not produce a PNG preview file.")

        return png_path.read_bytes()


def _resolve_pdftoppm() -> Path:
    env = os.environ.get("POPPLER_PDFTOPPM", "").strip()
    if env:
        candidate = Path(env).expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            return candidate

    which = shutil.which("pdftoppm")
    if which:
        return Path(which)

    for value in DEFAULT_PDFTOPPM_CANDIDATES:
        candidate = Path(value)
        if candidate.exists() and candidate.is_file():
            return candidate

    raise PdfPreviewError(
        "Poppler pdftoppm executable not found. Install Poppler and add pdftoppm to PATH, "
        "or set POPPLER_PDFTOPPM to the full executable path."
    )


def resolve_pdftoppm_status() -> tuple[bool, str, str]:
    env = os.environ.get("POPPLER_PDFTOPPM", "").strip()
    configured = env
    if env:
        candidate = Path(env).expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            return True, configured, str(candidate)

    which = shutil.which("pdftoppm")
    if which:
        return True, configured, str(Path(which).resolve())

    for value in DEFAULT_PDFTOPPM_CANDIDATES:
        candidate = Path(value)
        if candidate.exists() and candidate.is_file():
            return True, configured, str(candidate.resolve())

    return False, configured, ""


def set_pdftoppm_path(path: str) -> tuple[bool, str]:
    raw = path.strip()
    if not raw:
        return False, "Path is empty."
    candidate = Path(raw).expanduser().resolve()
    if not candidate.exists() or not candidate.is_file():
        return False, f"pdftoppm not found at: {candidate}"
    os.environ["POPPLER_PDFTOPPM"] = str(candidate)
    return True, str(candidate)
