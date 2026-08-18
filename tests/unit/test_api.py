"""
Unit tests for the DocsQuery FastAPI application.

These tests intentionally do NOT require Qdrant.

External services belong in integration tests.
"""

from fastapi.testclient import TestClient

from app.main import create_app
from app.retrieval.models import RetrievalResult


class FakeRetrievalService:
    """
    Fake retrieval service used by API unit tests.

    This prevents the tests from connecting to Qdrant.
    """

    def search(
        self,
        query: str,
        limit: int = 10,
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
                score=0.95,
            )
        ][:limit]


def create_test_client() -> TestClient:
    """
    Create a FastAPI test client using a fake retrieval
    service for the entire application lifecycle.

    This prevents the unit tests from creating real BM25,
    embedding, or Qdrant dependencies.
    """

    app = create_app(retrieval_factory=FakeRetrievalService)

    # The API dependency still resolves through
    # get_retrieval_service, but the application state
    # now contains FakeRetrievalService.
    return TestClient(app)


def test_health_endpoint():
    """
    The health endpoint should report that the application
    is running.
    """

    with create_test_client() as client:
        response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {"status": "ok"}


def test_search_endpoint_returns_results():
    """
    The search endpoint should return retrieval results.

    Qdrant is NOT required because the retrieval dependency
    is replaced with FakeRetrievalService.
    """

    with create_test_client() as client:
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

    assert body["results"][0]["text"] == "Python is a programming language."


def test_search_rejects_empty_query():
    """
    FastAPI validation should reject an empty query.
    """

    with create_test_client() as client:
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

    with create_test_client() as client:
        response = client.post(
            "/api/v1/search",
            json={
                "query": "Python",
                "limit": 100,
            },
        )

    assert response.status_code == 422
