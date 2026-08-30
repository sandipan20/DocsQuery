"""
DocsQuery - FastAPI Application

This file creates the main FastAPI application and registers
all API routes.

Current endpoints:

    GET  /health
    GET  /ready
    POST /api/v1/search
    POST /api/v1/query
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import (
    unexpected_exception_handler,
)
from app.api.middleware import request_id_middleware
from app.api.v1.query import router as query_router
from app.api.v1.search import router as search_router
from app.config.settings import get_settings
from app.container import AppContainer
from app.logging_config import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    Startup:
        - Create application dependencies.
        - Load persistent retrieval indexes.

    Shutdown:
        - Reserved for future cleanup.
    """

    # Create long-lived application dependencies once.
    container = AppContainer()

    # Restore the persistent BM25 index.
    loaded_chunks = container.load_indexes()

    # Store the container on the FastAPI application.
    # API routes will access it through request.app.state.
    app.state.container = container

    print(f"DocsQuery startup complete. Loaded {loaded_chunks} BM25 chunks.")

    yield

    print("DocsQuery shutting down.")


def create_app() -> FastAPI:
    """
    Create and configure the DocsQuery FastAPI application.
    """

    app = FastAPI(
        title="DocsQuery",
        description="Production-oriented domain-specific RAG API.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # --------------------------------------------------------
    # Liveness endpoint
    # --------------------------------------------------------
    # Answers:
    # "Is the HTTP application process alive?"
    # --------------------------------------------------------

    @app.get(
        "/health",
        tags=["health"],
    )
    def health() -> dict[str, str]:
        """
        Return basic application health.
        """

        return {
            "status": "ok",
        }

    # --------------------------------------------------------
    # Readiness endpoint
    # --------------------------------------------------------
    # Answers:
    # "Is the application ready to serve requests?"
    # --------------------------------------------------------

    @app.get(
        "/ready",
        tags=["health"],
    )
    def ready() -> dict:
        """
        Readiness endpoint.

        Returns 200 only when the application has the
        dependencies required to process requests.
        """

        container = app.state.container

        health = container.health_service.check()

        is_ready = all(health.values())

        if not is_ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "dependencies": health,
                },
            )

        return {
            "status": "ready",
            "dependencies": health,
        }

    # --------------------------------------------------------
    # Register API routers
    # --------------------------------------------------------

    # Search endpoint:
    #
    # POST /api/v1/search
    app.include_router(
        search_router,
        prefix="/api/v1",
    )

    # Query endpoint:
    #
    # POST /api/v1/query
    app.include_router(
        query_router,
        prefix="/api/v1",
    )

    app.middleware("http")(request_id_middleware)

    settings = get_settings()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    app.add_exception_handler(
        Exception,
        unexpected_exception_handler,
    )

    configure_logging()

    return app


# Create the application instance used by Uvicorn.
app = create_app()
