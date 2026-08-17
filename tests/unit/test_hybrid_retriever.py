"""
Unit tests for hybrid retrieval.
"""

from unittest.mock import MagicMock

import pytest

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import RetrievalResult


def result(
    chunk_id: str,
    score: float = 0.5,
) -> RetrievalResult:
    """
    Create a predictable retrieval result.
    """

    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-001",
        text=f"Text for {chunk_id}",
        source="test.pdf",
        page_number=1,
        chunk_index=0,
        score=score,
    )


def create_retriever():
    """
    Create a hybrid retriever with mocked backends.
    """

    bm25 = MagicMock()
    vector = MagicMock()

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=vector,
    )

    return hybrid, bm25, vector


def test_empty_query_is_rejected():
    """
    Empty queries should be rejected.
    """

    hybrid, _, _ = create_retriever()

    with pytest.raises(ValueError):
        hybrid.retrieve("")


def test_invalid_limit_is_rejected():
    """
    Final result limit must be positive.
    """

    hybrid, _, _ = create_retriever()

    with pytest.raises(ValueError):
        hybrid.retrieve(
            "Python",
            limit=0,
        )


def test_rrf_combines_results():
    """
    A document appearing in both retrieval systems should
    receive contributions from both rankings.
    """

    hybrid, bm25, vector = create_retriever()

    bm25.return_value = None

    bm25.retrieve.return_value = [
        result("chunk-A"),
        result("chunk-B"),
    ]

    vector.retrieve.return_value = [
        result("chunk-B"),
        result("chunk-C"),
    ]

    results = hybrid.retrieve(
        query="Python",
        limit=3,
        candidate_limit=2,
    )

    assert len(results) == 3

    # chunk-B appears at rank 2 in BM25 and rank 1
    # in vector retrieval, so it should receive contributions
    # from both systems.
    assert results[0].chunk_id == "chunk-B"


def test_duplicate_chunks_are_merged():
    """
    A chunk appearing in both result lists must appear only
    once in the final result list.
    """

    hybrid, bm25, vector = create_retriever()

    bm25.retrieve.return_value = [
        result("chunk-A"),
    ]

    vector.retrieve.return_value = [
        result("chunk-A"),
    ]

    results = hybrid.retrieve(
        query="Python",
        limit=10,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-A"


def test_rrf_score_is_positive():
    """
    Every returned result should have a positive RRF score.
    """

    hybrid, bm25, vector = create_retriever()

    bm25.retrieve.return_value = [
        result("chunk-A"),
    ]

    vector.retrieve.return_value = []

    results = hybrid.retrieve(
        query="Python",
    )

    assert results[0].score > 0
