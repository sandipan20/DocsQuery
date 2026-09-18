"""
Unit tests for the retrieval service.

These tests verify that the service correctly coordinates:

    Vector confidence gating
    Hybrid retrieval
    Cross-encoder reranking
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
    confidence_gate = MagicMock()

    # The BM25 index exposes its underlying retriever.
    bm25_index.retriever = MagicMock()

    # Vector retrieval returns the initial semantic candidates.
    vector_results = [
        create_result("chunk-001"),
        create_result("chunk-002"),
    ]

    # Hybrid retrieval returns candidate documents.
    candidates = [
        create_result("chunk-001"),
        create_result("chunk-002"),
    ]

    # The reranker returns the final results.
    final_results = [
        create_result("chunk-001"),
    ]

    vector_retriever.retrieve.return_value = vector_results
    confidence_gate.is_confident.return_value = True

    service = RetrievalService(
        bm25_index=bm25_index,
        vector_retriever=vector_retriever,
        reranker=reranker,
        confidence_gate=confidence_gate,
    )

    return (
        service,
        bm25_index,
        vector_retriever,
        reranker,
        confidence_gate,
        vector_results,
        candidates,
        final_results,
    )


def test_search_delegates_to_retrieval_pipeline():
    """
    The service should:

        1. Perform vector retrieval.
        2. Check vector confidence.
        3. Perform hybrid retrieval.
        4. Pass hybrid candidates to the reranker.
        5. Return reranked results.
    """

    (
        service,
        _,
        vector_retriever,
        reranker,
        confidence_gate,
        vector_results,
        candidates,
        final_results,
    ) = create_service()

    # Replace the internally-created hybrid retriever with
    # a mock so this test only checks service orchestration.
    service.hybrid_retriever = MagicMock()

    service.hybrid_retriever.retrieve.return_value = candidates
    reranker.rerank.return_value = final_results

    results = service.search(
        query="What is Python?",
        limit=1,
    )

    # Verify vector retrieval was called first.
    vector_retriever.retrieve.assert_called_once_with(
        query="What is Python?",
        limit=20,
    )

    # Verify the confidence gate received vector results.
    confidence_gate.is_confident.assert_called_once_with(
        vector_results,
    )

    # Verify hybrid retrieval was called only after
    # the confidence check passed.
    service.hybrid_retriever.retrieve.assert_called_once_with(
        query="What is Python?",
        limit=20,
        candidate_limit=20,
        vector_results=vector_results,
    )

    # Verify the candidates were passed to the reranker.
    reranker.rerank.assert_called_once_with(
        query="What is Python?",
        results=candidates,
        top_k=1,
    )

    assert results == final_results


def test_search_abstains_when_vector_confidence_is_low():
    """
    The service should stop retrieval when the vector similarity
    confidence gate rejects the query.
    """

    (
        service,
        _,
        vector_retriever,
        reranker,
        confidence_gate,
        vector_results,
        _,
        _,
    ) = create_service()

    # The confidence gate rejects the vector results.
    confidence_gate.is_confident.return_value = False

    service.hybrid_retriever = MagicMock()

    results = service.search(
        query="This query is outside the indexed corpus.",
        limit=5,
    )

    # Vector retrieval should still happen because the gate
    # needs its top result to evaluate confidence.
    vector_retriever.retrieve.assert_called_once_with(
        query="This query is outside the indexed corpus.",
        limit=20,
    )

    confidence_gate.is_confident.assert_called_once_with(
        vector_results,
    )

    # Low confidence means no hybrid retrieval.
    service.hybrid_retriever.retrieve.assert_not_called()

    # No reranking should happen either.
    reranker.rerank.assert_not_called()

    # The service abstains by returning no results.
    assert results == []


def test_retriever_errors_are_propagated():
    """
    Retrieval errors should not be silently swallowed.

    Infrastructure and validation failures must reach the
    API layer so they can be handled correctly.
    """

    (
        service,
        _,
        vector_retriever,
        _,
        _,
        _,
        _,
        _,
    ) = create_service()

    vector_retriever.retrieve.side_effect = ValueError(
        "Query cannot be empty.",
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty.",
    ):
        service.search(
            query="Python",
            limit=5,
        )
