"""LeakLens backend application entry-point."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .core.config import settings
from .schemas import HealthResponse


def create_app() -> FastAPI:
    app = FastAPI(
        title="LeakLens API",
        description="Budget-aware adaptive data-leakage investigation for ML datasets.",
        version=__version__,
    )

    # CORS — allow the Vite dev server during development.
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            app="leaklens",
            version=__version__,
            gemini_configured=bool(settings.gemini_api_key),
        )

    # Safety: refuse to bind to the network without authentication.
    from .core.security import assert_safe_binding
    assert_safe_binding(settings.host, settings.api_key)

    # API routers are registered here as they are built out.
    from .api import investigations as investigations_api
    app.include_router(investigations_api.router, prefix="/api")

    return app


app = create_app()
