from __future__ import annotations

from fastapi import FastAPI

from backend.api.routers.chat import router as chat_router
from backend.api.routers.export import router as export_router
from backend.api.routers.loaders import router as loaders_router
from backend.api.routers.retrieval import router as retrieval_router
from backend.dependencies import close_runtime


def create_app() -> FastAPI:
    app = FastAPI(title="Python Workflow API", version="1.0.0")

    app.include_router(loaders_router)
    app.include_router(retrieval_router)
    app.include_router(chat_router)
    app.include_router(export_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.on_event("shutdown")
    def _shutdown() -> None:
        close_runtime()

    return app


app = create_app()
