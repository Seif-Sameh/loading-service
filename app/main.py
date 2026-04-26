"""FastAPI application entry point.

Run with::

    uvicorn app.main:app --host 0.0.0.0 --port 8009 --reload
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import catalog_router, solve_router
from app.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="Alexandria Port — loading-service",
        description="Container-loading optimisation backend (heuristics + GA + PPO+Transformer).",
        version=__version__,
    )
    # During dev we let the frontend dev server hit us from any localhost origin. In
    # production this comes from an env var; the api-gateway in the parent project
    # handles real CORS/auth.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(catalog_router)
    app.include_router(solve_router)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host=settings.api_host, port=settings.api_port, reload=False
    )
