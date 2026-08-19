"""
DocsQuery - FastAPI Application

Main HTTP application.

Current endpoints:

    GET  /health
    POST /api/v1/search
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.api.v1.search import router as search_router
from app.container import AppContainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    Startup:
        - Create application dependencies.
        - Load persistent indexes.

    Shutdown:
        - Future cleanup operations will go here.
    """

    # --------------------------------------------------------
    # Startup
    # --------------------------------------------------------

    container = AppContainer()

    loaded_chunks = container.load_indexes()

    # Store the container on the FastAPI application object.
    app.state.container = container

    print(f"DocsQuery startup complete. Loaded {loaded_chunks} BM25 chunks.")

    yield

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    print("DocsQuery shutting down.")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """

    app = FastAPI(
        title="DocsQuery",
        description=("Production-oriented domain-specific RAG API."),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get(
        "/health",
        tags=["health"],
    )
    def health() -> dict[str, str]:
        """
        Basic liveness endpoint.
        """

        return {
            "status": "ok",
        }

    @app.get(
        "/ready",
        tags=["health"],
    )
    def ready() -> dict[str, str]:
        """
        Readiness endpoint.

        Indicates whether the application has loaded the
        retrieval index required to serve searches.
        """

        container = app.state.container

        if not container.bm25_loaded:
            raise HTTPException(
                status_code=503,
                detail="Retrieval indexes are not ready.",
            )

        return {
            "status": "ready",
        }

    app.include_router(
        search_router,
        prefix="/api/v1",
    )

    return app


app = create_app()
