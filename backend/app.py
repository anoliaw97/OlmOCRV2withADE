from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routers.chat import router as chat_router
from backend.api.routers.export import router as export_router
from backend.api.routers.loaders import router as loaders_router
from backend.api.routers.retrieval import router as retrieval_router
from backend.api.routers.system import router as system_router
from backend.dependencies import close_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_ROOT = PROJECT_ROOT / "webapp"
ASSETS_ROOT = WEBAPP_ROOT / "assets"


def create_app() -> FastAPI:
    app = FastAPI(title="Python Workflow WebApp API", version="1.1.0")

    app.include_router(system_router)
    app.include_router(loaders_router)
    app.include_router(retrieval_router)
    app.include_router(chat_router)
    app.include_router(export_router)

    app.mount("/assets", StaticFiles(directory=str(ASSETS_ROOT)), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(WEBAPP_ROOT / "index.html"))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.on_event("shutdown")
    def _shutdown() -> None:
        close_runtime()

    return app


app = create_app()
