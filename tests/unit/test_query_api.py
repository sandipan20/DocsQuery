"""
Unit tests for the RAG query API.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.generation.models import GeneratedAnswer
from app.main import create_app
from app.retrieval.models import RetrievalResult
from app.services.rag_service import (
    InsufficientEvidenceError,
)


def create_result() -> RetrievalResult:
    """
    Create predictable evidence for API tests.
    """

    return RetrievalResult(
        chunk_id="chunk-001",
        document_id="doc-001",
        text="Python is a programming language.",
        source="python.pdf",
        page_number=1,
        chunk_index=0,
        score=0.9,
    )


def test_query_returns_answer_and_citation():
    """
    The query endpoint should return a grounded answer,
    citation metadata, and RAG performance metrics.
    """

    app = create_app()

    with TestClient(app) as client:
        # Replace the real RAG service with a mock so this
        # unit test does not call Qdrant, the reranker, or Gemini.
        app.state.container.rag_service = MagicMock()

        # Create a predictable fake RAG response.
        app.state.container.rag_service.query.return_value = MagicMock(
            query="What is Python?",
            generated_answer=GeneratedAnswer(
                answer=("Python is a programming language. [C1]"),
                citations=["C1"],
            ),
            results=[create_result()],
            # Mock RAG performance metrics.
            retrieval_latency_ms=120.0,
            generation_latency_ms=500.0,
            total_latency_ms=620.0,
        )

        # Send the HTTP request to the API.
        response = client.post(
            "/api/v1/query",
            json={
                "query": "What is Python?",
                "top_k": 5,
            },
        )

    # Verify successful HTTP response.
    assert response.status_code == 200

    body = response.json()

    # Verify generated answer.
    assert body["answer"] == "Python is a programming language. [C1]"

    # Verify citation metadata.
    assert body["citations"] == [
        {
            "citation_id": "C1",
            "source": "python.pdf",
            "page_number": 1,
            "chunk_id": "chunk-001",
        }
    ]

    # Verify RAG performance metrics.
    assert body["metrics"]["retrieval_latency_ms"] == 120.0
    assert body["metrics"]["generation_latency_ms"] == 500.0
    assert body["metrics"]["total_latency_ms"] == 620.0


def test_query_returns_404_when_evidence_is_missing():
    """
    The API should return 404 when no relevant evidence
    is available.
    """

    app = create_app()

    with TestClient(app) as client:
        # Replace the real RAG service with a mock.
        app.state.container.rag_service = MagicMock()

        # Simulate insufficient retrieval evidence.
        app.state.container.rag_service.query.side_effect = InsufficientEvidenceError(
            "No relevant evidence was found."
        )

        response = client.post(
            "/api/v1/query",
            json={
                "query": "Unknown question",
                "top_k": 5,
            },
        )

    assert response.status_code == 404


def test_query_rejects_empty_query():
    """
    The API should reject an empty query.
    """

    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/query",
            json={
                "query": "",
                "top_k": 5,
            },
        )

    assert response.status_code == 422


def test_query_rejects_invalid_top_k():
    """
    The API should reject a top_k value outside the
    allowed range.
    """

    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/query",
            json={
                "query": "Python",
                "top_k": 100,
            },
        )

    assert response.status_code == 422
