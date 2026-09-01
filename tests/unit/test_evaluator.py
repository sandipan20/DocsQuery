"""
Tests for the retrieval evaluator.
"""

import pytest

from app.evaluation.evaluator import (
    RetrievalEvaluator,
)
from app.evaluation.models import (
    EvaluationExample,
)


def create_examples():
    """
    Create a small deterministic evaluation dataset.
    """

    return [
        EvaluationExample(
            example_id="q001",
            query="query one",
            relevant_chunk_ids=["chunk-a"],
        ),
        EvaluationExample(
            example_id="q002",
            query="query two",
            relevant_chunk_ids=["chunk-b"],
        ),
    ]


def fake_retriever(
    query: str,
    limit: int,
) -> list[str]:
    """
    Deterministic fake retriever for testing.
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
    Verify that the evaluator produces a valid result.
    """

    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        strategy="fake",
        examples=create_examples(),
        retriever=fake_retriever,
    )

    assert result.strategy == "fake"
    assert result.num_examples == 2

    # Query 1 retrieves the relevant chunk at rank 1.
    # Query 2 retrieves it at rank 2.
    #
    # Therefore:
    # MRR = (1 + 0.5) / 2 = 0.75
    assert result.metrics.mrr == pytest.approx(0.75)

    assert result.metrics.recall_at_1 == pytest.approx(0.5)

    assert result.metrics.recall_at_3 == pytest.approx(1.0)


def test_empty_dataset_is_rejected():
    """
    Evaluating no examples should fail clearly.
    """

    evaluator = RetrievalEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            strategy="fake",
            examples=[],
            retriever=fake_retriever,
        )
