"""
Tests for the retrieval evaluator.

The evaluator receives a list of evaluation examples and a
retrieval function, then calculates aggregate retrieval metrics.

These tests use a deterministic fake retriever so no external
services such as Qdrant or an embedding model are required.
"""

from builtins import ValueError

import pytest

from app.evaluation.evaluator import RetrievalEvaluator
from app.evaluation.models import EvaluationExample


def create_examples():
    """
    Create a small deterministic evaluation dataset.

    query_type and difficulty are required because
    EvaluationExample stores both the category and difficulty
    of each evaluation question.
    """

    return [
        EvaluationExample(
            example_id="q001",
            query="query one",
            query_type="test",
            difficulty="easy",
            relevant_chunk_ids=["chunk-a"],
        ),
        EvaluationExample(
            example_id="q002",
            query="query two",
            query_type="test",
            difficulty="easy",
            relevant_chunk_ids=["chunk-b"],
        ),
    ]


def fake_retriever(
    query: str,
    limit: int,
) -> list[str]:
    """
    Deterministic fake retriever used only for testing.

    Query one finds its relevant chunk at rank 1.

    Query two finds its relevant chunk at rank 10.

    This verifies that the evaluator requests enough candidates
    to calculate Recall@10 and Recall@20.
    """

    if query == "query one":
        results = [
            "chunk-a",
            "chunk-x1",
            "chunk-x2",
            "chunk-x3",
            "chunk-x4",
            "chunk-x5",
            "chunk-x6",
            "chunk-x7",
            "chunk-x8",
            "chunk-x9",
            "chunk-x10",
        ]

        return results[:limit]

    results = [
        "chunk-x1",
        "chunk-x2",
        "chunk-x3",
        "chunk-x4",
        "chunk-x5",
        "chunk-x6",
        "chunk-x7",
        "chunk-x8",
        "chunk-x9",
        "chunk-b",
        "chunk-x11",
        "chunk-x12",
        "chunk-x13",
        "chunk-x14",
        "chunk-x15",
        "chunk-x16",
        "chunk-x17",
        "chunk-x18",
        "chunk-x19",
        "chunk-x20",
    ]

    return results[:limit]


def test_evaluator_calculates_metrics():
    """
    Verify that the evaluator calculates retrieval metrics
    correctly for a deterministic retriever.
    """

    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        strategy="fake",
        examples=create_examples(),
        retriever=fake_retriever,
    )

    assert result.strategy == "fake"
    assert result.num_examples == 2

    # Query 1:
    #
    #   chunk-a is ranked first
    #   Reciprocal Rank = 1 / 1 = 1.0
    #
    # Query 2:
    #
    #   chunk-b is ranked tenth
    #   It is outside the top-5 window.
    #
    # MRR intentionally remains a top-5 metric here so the
    # previous benchmark semantics are preserved.
    #
    # Therefore:
    #
    #   MRR = (1.0 + 0.0) / 2
    #       = 0.5
    assert result.metrics.mrr == pytest.approx(0.5)

    # At rank 1, only the first query finds its relevant
    # document.
    #
    # Recall@1 = 1 / 2 = 0.5
    assert result.metrics.recall_at_1 == pytest.approx(0.5)

    # At rank 3, only the first query has found its relevant
    # document because chunk-b is ranked tenth.
    assert result.metrics.recall_at_3 == pytest.approx(0.5)

    # At rank 5, the second query still has not found chunk-b.
    assert result.metrics.recall_at_5 == pytest.approx(0.5)

    # At rank 10, both queries have found their relevant
    # documents.
    assert result.metrics.recall_at_10 == pytest.approx(1.0)

    # The relevant result for query two remains within the
    # requested top-20 candidate window.
    assert result.metrics.recall_at_20 == pytest.approx(1.0)


def test_empty_dataset_is_rejected():
    """
    Evaluating an empty dataset should fail clearly.
    """

    evaluator = RetrievalEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            strategy="fake",
            examples=[],
            retriever=fake_retriever,
        )
