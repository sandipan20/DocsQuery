"""
Unit tests for the DocsQuery FastAPI application.
"""

from fastapi.testclient import TestClient

from app.main import create_app
from app.retrieval.models import RetrievalResult


class FakeRetrievalService:
    """
    Fake retrieval service used by API tests.

    This avoids requiring Qdrant or an embedding model
    during API unit tests.
    """

    def search(
        self,
        query: str,
        limit: int,
    ) -> list[RetrievalResult]:
        """
        Return predictable fake retrieval results.
        """

        return [
            RetrievalResult(
                chunk_id="chunk-001",
                document_id="doc-001",
                text="Python is a programming language.",
                source="python.pdf",
                page_number=1,
                chunk_index=0,
                score=0.032,
            )
        ][:limit]


def create_test_app():
    """
    Create a FastAPI application with a fake retrieval service.
    """

    app = create_app()

    # The lifespan hasn't run when TestClient is created
    # until the context manager starts.
    return app


def test_health_endpoint():
    """
    The health endpoint should report that the application
    is running.
    """

    app = create_test_app()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {"status": "ok"}


def test_search_endpoint_returns_results():
    """
    The search endpoint should return retrieval results.
    """

    app = create_test_app()

    with TestClient(app) as client:
        app.state.container.retrieval_service = FakeRetrievalService()

        response = client.post(
            "/api/v1/search",
            json={
                "query": "What is Python?",
                "limit": 5,
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["query"] == "What is Python?"
    assert len(body["results"]) == 1

    assert body["results"][0]["chunk_id"] == "chunk-001"


def test_search_rejects_empty_query():
    """
    FastAPI validation should reject an empty query.
    """

    app = create_test_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/search",
            json={
                "query": "",
                "limit": 5,
            },
        )

    assert response.status_code == 422


def test_search_rejects_invalid_limit():
    """
    The API should reject limits outside the allowed range.
    """

    app = create_test_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/search",
            json={
                "query": "Python",
                "limit": 100,
            },
        )

    assert response.status_code == 422
