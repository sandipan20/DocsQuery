"""
Unit tests for evaluation models.
"""

import pytest

from app.evaluation.models import EvaluationExample


def test_positive_example_requires_relevant_chunk():
    """
    Positive examples must contain at least one relevant chunk.
    """

    example = EvaluationExample(
        example_id="q001",
        query="What does git init do?",
        query_type="repository-basics",
        difficulty="easy",
        relevant_chunk_ids=["chunk-001"],
    )

    assert example.difficulty == "easy"
    assert example.relevant_chunk_ids == ["chunk-001"]


def test_negative_example_allows_empty_relevance():
    """
    Negative examples may intentionally have zero relevant chunks.
    """

    example = EvaluationExample(
        example_id="neg001",
        query="How do I configure Kubernetes autoscaling?",
        query_type="outside-corpus",
        difficulty="negative",
        relevant_chunk_ids=[],
    )

    assert example.difficulty == "negative"
    assert example.relevant_chunk_ids == []


def test_negative_example_with_relevance_fails():
    """
    Negative examples cannot have relevant chunks.
    """

    with pytest.raises(ValueError):
        EvaluationExample(
            example_id="neg002",
            query="How do I configure Kubernetes autoscaling?",
            query_type="outside-corpus",
            difficulty="negative",
            relevant_chunk_ids=["chunk-001"],
        )


def test_positive_example_without_relevance_fails():
    """
    Non-negative examples cannot have an empty relevance set.
    """

    with pytest.raises(ValueError):
        EvaluationExample(
            example_id="q002",
            query="How do I create a Git branch?",
            query_type="branching",
            difficulty="medium",
            relevant_chunk_ids=[],
        )
