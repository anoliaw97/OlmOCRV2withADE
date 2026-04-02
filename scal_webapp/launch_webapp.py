from __future__ import annotations

import importlib
import sys
import traceback


REQUIRED_MODULES = [
    "fastapi",
    "uvicorn",
    "jinja2",
    "sqlalchemy",
    "pandas",
    "openpyxl",
    "docx",
    "pypdf",
    "sklearn",
    "joblib",
]


def _check_dependencies() -> list[str]:
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    return missing


def main() -> int:
    missing = _check_dependencies()
    if missing:
        print("Missing dependencies detected:")
        for m in missing:
            print(f" - {m}")
        print("\nInstall them with:")
        print("  pip install -r scal_webapp/requirements.txt")
        return 1

    try:
        import uvicorn

        print("Starting SCAL web app on http://localhost:8080")
        uvicorn.run("scal_webapp.backend.main:app", host="0.0.0.0", port=8080, reload=False)
        return 0
    except Exception as exc:
        print(f"Failed to start web app: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
