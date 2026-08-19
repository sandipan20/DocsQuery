"""
Unit tests for the cross-encoder reranker.
"""

import pytest

from app.retrieval.models import RetrievalResult
from app.retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """
    Fake cross-encoder used for unit testing.

    It returns predictable relevance scores without
    downloading a real model.
    """

    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        """
        Return predefined scores.
        """

        assert len(pairs) == len(self.scores)

        return self.scores


def create_result(
    chunk_id: str,
    text: str,
) -> RetrievalResult:
    """
    Create a predictable retrieval result.
    """

    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-001",
        text=text,
        source="test.pdf",
        page_number=1,
        chunk_index=0,
        score=0.1,
    )


def create_reranker(
    scores,
) -> CrossEncoderReranker:
    """
    Create a reranker without loading a real model.
    """

    reranker = object.__new__(CrossEncoderReranker)

    reranker.model_name = "fake-model"

    reranker.model = FakeCrossEncoder(scores)

    return reranker


def test_empty_query_is_rejected():
    """
    Empty queries should not be reranked.
    """

    reranker = create_reranker([])

    with pytest.raises(ValueError):
        reranker.rerank(
            "",
            [],
        )


def test_invalid_top_k_is_rejected():
    """
    top_k must be positive.
    """

    reranker = create_reranker([])

    results = [
        create_result(
            "chunk-1",
            "Python",
        )
    ]

    with pytest.raises(ValueError):
        reranker.rerank(
            "What is Python?",
            results,
            top_k=0,
        )


def test_empty_results_return_empty():
    """
    Empty candidate lists should return immediately.
    """

    reranker = create_reranker([])

    results = reranker.rerank(
        "What is Python?",
        [],
    )

    assert results == []


def test_results_are_sorted_by_reranker_score():
    """
    Results should be sorted by descending cross-encoder
    relevance score.
    """

    reranker = create_reranker(
        [
            0.20,
            0.95,
            0.50,
        ]
    )

    results = [
        create_result(
            "chunk-1",
            "Python programming",
        ),
        create_result(
            "chunk-2",
            "Python language",
        ),
        create_result(
            "chunk-3",
            "Programming language",
        ),
    ]

    reranked = reranker.rerank(
        "What is Python?",
        results,
        top_k=3,
    )

    assert [result.chunk_id for result in reranked] == [
        "chunk-2",
        "chunk-3",
        "chunk-1",
    ]

    assert reranked[0].score == 0.95


def test_top_k_is_respected():
    """
    Reranking should return no more than top_k results.
    """

    reranker = create_reranker(
        [
            0.9,
            0.8,
            0.7,
        ]
    )

    results = [
        create_result(
            "chunk-1",
            "Python",
        ),
        create_result(
            "chunk-2",
            "Qdrant",
        ),
        create_result(
            "chunk-3",
            "BM25",
        ),
    ]

    reranked = reranker.rerank(
        "Python",
        results,
        top_k=2,
    )

    assert len(reranked) == 2
