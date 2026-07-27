"""FastAPI application exposing the intrusion detection service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api import state as service_state
from src.api.routers import detection, health
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.validators import ValidationError

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model on start-up and release it on shutdown."""
    service_state.initialise(get_config())
    try:
        yield
    finally:
        service_state.shutdown()


def create_app() -> FastAPI:
    """Build the configured FastAPI application."""
    config = get_config()
    app = FastAPI(
        title=config.app.name,
        version=config.app.version,
        description="Machine learning network intrusion detection and alerting service.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
        log.warning("Validation error on {}: {}", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc), "error_type": "ValidationError"},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error on {}", request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "error_type": type(exc).__name__},
        )

    app.include_router(health.router)
    app.include_router(detection.router)

    @app.get("/", tags=["health"], summary="Service banner")
    def root() -> dict[str, str]:
        return {
            "service": config.app.name,
            "version": config.app.version,
            "docs": "/docs",
        }

    return app


app = create_app()


def run() -> None:
    """Start the service with uvicorn using the configured host and port."""
    import uvicorn

    config = get_config()
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
    )


if __name__ == "__main__":
    run()
