"""
DocsQuery - FastAPI Application

Main HTTP application.

The application lifecycle is responsible for creating expensive
shared dependencies once at startup.
"""

from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI

from app.api.v1.search import router as search_router
from app.core.app_state import AppState
from app.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval_service import RetrievalService


def create_retrieval_service() -> RetrievalService:
    """
    Build the production retrieval service.

    This function is intentionally separate from the FastAPI
    application so tests can replace it with a fake service.

    The exact construction of HybridRetriever is kept here
    rather than inside an HTTP route.
    """

    retriever = HybridRetriever()

    return RetrievalService(
        retriever=retriever,
    )


def create_lifespan(
    retrieval_factory: Callable[[], RetrievalService] = create_retrieval_service,
):
    """
    Create the FastAPI lifespan handler.

    Args:
        retrieval_factory:
            Function used to construct the application-wide
            RetrievalService.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        Initialize and clean up application dependencies.
        """

        # ----------------------------------------------------
        # Startup
        # ----------------------------------------------------

        retrieval_service = retrieval_factory()

        app.state.docsquery = AppState(retrieval_service=retrieval_service)

        yield

        # ----------------------------------------------------
        # Shutdown
        # ----------------------------------------------------

        app.state.docsquery = None

    return lifespan


def create_app(
    retrieval_factory: Callable[[], RetrievalService] = create_retrieval_service,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    The retrieval factory can be replaced during testing.
    """

    app = FastAPI(
        title="DocsQuery",
        description=("Production-oriented domain-specific RAG API."),
        version="0.1.0",
        lifespan=create_lifespan(retrieval_factory),
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

    app.include_router(
        search_router,
        prefix="/api/v1",
    )

    return app


app = create_app()
