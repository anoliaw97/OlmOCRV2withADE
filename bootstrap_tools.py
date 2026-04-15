from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


POPPLER_RELEASE_URLS = [
    "https://github.com/oschwartz10612/poppler-windows/releases/download/v24.08.0-0/Release-24.08.0-0.zip",
    "https://github.com/oschwartz10612/poppler-windows/releases/download/v23.11.0-0/Release-23.11.0-0.zip",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap local Windows tools (Poppler) for Workflow WebApp")
    parser.add_argument("--no-setx", action="store_true", help="Do not persist environment variables with setx")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    tools_dir = project_root / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    print("[setup] Checking Poppler (pdftoppm)...")
    pdftoppm_path = resolve_pdftoppm(project_root)
    if pdftoppm_path:
        print(f"[setup] Poppler already available: {pdftoppm_path}")
    else:
        pdftoppm_path = install_poppler(tools_dir)
        if not pdftoppm_path:
            print("[setup] Poppler install failed. Please install manually.")
            return 1

    os.environ["POPPLER_PDFTOPPM"] = str(pdftoppm_path)
    print(f"[setup] Set POPPLER_PDFTOPPM for current shell: {pdftoppm_path}")

    if not args.no_setx:
        persist_env_var("POPPLER_PDFTOPPM", str(pdftoppm_path))

    llama_cli = shutil.which("llama-cli") or shutil.which("llama-cli.exe")
    if llama_cli:
        print(f"[setup] Found llama-cli on PATH: {llama_cli}")
    else:
        print(
            "[setup] llama-cli was not found on PATH.\n"
            "        If you use llama.cpp backend, set full path in UI, for example:\n"
            "        C:\\llama.cpp\\build\\bin\\Release\\llama-cli.exe"
        )

    print("[setup] Done.")
    return 0


def resolve_pdftoppm(project_root: Path) -> Path | None:
    env = os.environ.get("POPPLER_PDFTOPPM", "").strip()
    if env:
        candidate = Path(env)
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    which = shutil.which("pdftoppm")
    if which:
        return Path(which).resolve()

    local_candidate = project_root / "tools" / "poppler" / "Library" / "bin" / "pdftoppm.exe"
    if local_candidate.exists() and local_candidate.is_file():
        return local_candidate.resolve()

    return None


def install_poppler(tools_dir: Path) -> Path | None:
    poppler_dir = tools_dir / "poppler"
    poppler_dir.mkdir(parents=True, exist_ok=True)

    archive_path = tools_dir / "poppler_windows.zip"
    for url in POPPLER_RELEASE_URLS:
        print(f"[setup] Downloading Poppler from: {url}")
        try:
            urllib.request.urlretrieve(url, archive_path)
            break
        except Exception as exc:
            print(f"[setup] Download failed: {exc}")
    else:
        return None

    try:
        with zipfile.ZipFile(archive_path, "r") as zipf:
            tmp_extract = tools_dir / "_poppler_extract"
            if tmp_extract.exists():
                shutil.rmtree(tmp_extract, ignore_errors=True)
            tmp_extract.mkdir(parents=True, exist_ok=True)
            zipf.extractall(tmp_extract)

        extracted_root = None
        for child in tmp_extract.iterdir():
            if child.is_dir():
                extracted_root = child
                break
        if not extracted_root:
            extracted_root = tmp_extract

        if poppler_dir.exists():
            shutil.rmtree(poppler_dir, ignore_errors=True)
        shutil.copytree(extracted_root, poppler_dir)
    except Exception as exc:
        print(f"[setup] Extraction failed: {exc}")
        return None
    finally:
        try:
            archive_path.unlink(missing_ok=True)
        except Exception:
            pass

    pdftoppm = poppler_dir / "Library" / "bin" / "pdftoppm.exe"
    if pdftoppm.exists() and pdftoppm.is_file():
        print(f"[setup] Poppler installed at: {pdftoppm}")
        return pdftoppm.resolve()

    print("[setup] Installed archive did not contain pdftoppm.exe in expected location.")
    return None


def persist_env_var(name: str, value: str) -> None:
    if os.name != "nt":
        return
    try:
        import subprocess

        subprocess.run(["setx", name, value], check=False, capture_output=True, text=True)
        print(f"[setup] Persisted with setx: {name}")
    except Exception as exc:
        print(f"[setup] Could not run setx for {name}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
