from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .database import Base, engine
from .routers.chat import router as chat_router
from .routers.extraction import router as extraction_router
from .routers.ui import router as ui_router


def create_app() -> FastAPI:
    app = FastAPI(title="SCAL Extraction + Offline RAG", version="0.1.0")
    Base.metadata.create_all(bind=engine)

    app.mount("/static", StaticFiles(directory="scal_webapp/backend/static"), name="static")
    app.include_router(ui_router)
    app.include_router(extraction_router)
    app.include_router(chat_router)
    return app


app = create_app()
