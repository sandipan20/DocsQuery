"""
Tests for the retrieval evaluator.

The evaluator receives a list of evaluation examples and a
retrieval function, then calculates aggregate retrieval metrics.

These tests use a deterministic fake retriever so no external
services such as Qdrant or an embedding model are required.
"""

import pytest

from app.evaluation.evaluator import RetrievalEvaluator
from app.evaluation.models import EvaluationExample


def create_examples():
    """
    Create a small deterministic evaluation dataset.

    query_type is required because EvaluationExample now stores
    the category of each evaluation question.
    """

    return [
        EvaluationExample(
            example_id="q001",
            query="query one",
            query_type="test",
            relevant_chunk_ids=["chunk-a"],
        ),
        EvaluationExample(
            example_id="q002",
            query="query two",
            query_type="test",
            relevant_chunk_ids=["chunk-b"],
        ),
    ]


def fake_retriever(
    query: str,
    limit: int,
) -> list[str]:
    """
    Deterministic fake retriever used only for testing.

    The results are intentionally predictable so that we can
    calculate the expected MRR and Recall values by hand.
    """

    if query == "query one":
        return [
            "chunk-a",
            "chunk-x",
        ][:limit]

    return [
        "chunk-x",
        "chunk-b",
    ][:limit]


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
    #   chunk-b is ranked second
    #   Reciprocal Rank = 1 / 2 = 0.5
    #
    # Therefore:
    #
    #   MRR = (1.0 + 0.5) / 2
    #       = 0.75
    assert result.metrics.mrr == pytest.approx(0.75)

    # At rank 1, only the first query finds its relevant
    # document.
    #
    # Recall@1 = 1 / 2 = 0.5
    assert result.metrics.recall_at_1 == pytest.approx(0.5)

    # At rank 3, both queries have found their relevant
    # documents.
    #
    # Recall@3 = 2 / 2 = 1.0
    assert result.metrics.recall_at_3 == pytest.approx(1.0)


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
