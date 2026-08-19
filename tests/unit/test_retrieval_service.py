"""
Unit tests for the retrieval service.

These tests verify that the service correctly coordinates
hybrid retrieval and cross-encoder reranking.
"""

from unittest.mock import MagicMock

import pytest

from app.retrieval.models import RetrievalResult
from app.services.retrieval_service import RetrievalService


def create_result(
    chunk_id: str = "chunk-001",
) -> RetrievalResult:
    """
    Create a predictable retrieval result for testing.
    """

    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-001",
        text="Python is a programming language.",
        source="test.pdf",
        page_number=1,
        chunk_index=0,
        score=0.5,
    )


def create_service():
    """
    Create a RetrievalService with mocked dependencies.

    This is important because unit tests should not require:

        - Qdrant
        - embedding models
        - cross-encoder models
        - external services
    """

    bm25_index = MagicMock()
    vector_retriever = MagicMock()
    reranker = MagicMock()

    # The BM25 index exposes its underlying retriever.
    bm25_index.retriever = MagicMock()

    # Hybrid retrieval returns candidate documents.
    candidates = [
        create_result("chunk-001"),
        create_result("chunk-002"),
    ]

    # The reranker returns the final results.
    final_results = [
        create_result("chunk-001"),
    ]

    return (
        RetrievalService(
            bm25_index=bm25_index,
            vector_retriever=vector_retriever,
            reranker=reranker,
        ),
        bm25_index,
        vector_retriever,
        reranker,
        candidates,
        final_results,
    )


def test_search_delegates_to_retrieval_pipeline():
    """
    The service should delegate retrieval to the configured
    hybrid retriever and then pass the candidates to the
    reranker.
    """

    (
        service,
        _,
        _,
        reranker,
        candidates,
        final_results,
    ) = create_service()

    # Replace the internally-created hybrid retriever with
    # a mock so this test only checks service orchestration.
    service.hybrid_retriever = MagicMock()

    service.hybrid_retriever.retrieve.return_value = (
        candidates
    )

    reranker.rerank.return_value = final_results

    results = service.search(
        query="What is Python?",
        limit=1,
    )

    # Verify hybrid retrieval was called.
    service.hybrid_retriever.retrieve.assert_called_once_with(
        query="What is Python?",
        limit=20,
        candidate_limit=20,
    )

    # Verify the candidates were passed to the reranker.
    reranker.rerank.assert_called_once_with(
        query="What is Python?",
        results=candidates,
        top_k=1,
    )

    assert results == final_results


def test_retriever_errors_are_propagated():
    """
    Retrieval errors should not be silently swallowed.

    Infrastructure and validation failures must reach the
    API layer so they can be handled correctly.
    """

    service, _, _, _, _, _ = create_service()

    service.hybrid_retriever = MagicMock()

    service.hybrid_retriever.retrieve.side_effect = (
        ValueError("Query cannot be empty.")
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty.",
    ):
        service.search(
            query="Python",
            limit=5,
        )