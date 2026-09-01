"""
Tests for retrieval evaluation metrics.
"""

import pytest

from app.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    """
    Two relevant chunks found out of two means perfect recall.
    """

    retrieved = [
        "chunk-a",
        "chunk-c",
        "chunk-x",
    ]

    relevant = {
        "chunk-a",
        "chunk-c",
    }

    assert (
        recall_at_k(
            retrieved,
            relevant,
            k=3,
        )
        == 1.0
    )


def test_partial_recall_at_k():
    """
    Finding one of two relevant chunks gives 0.5 recall.
    """

    retrieved = [
        "chunk-a",
        "chunk-x",
    ]

    relevant = {
        "chunk-a",
        "chunk-c",
    }

    assert (
        recall_at_k(
            retrieved,
            relevant,
            k=2,
        )
        == 0.5
    )


def test_precision_at_k():
    """
    Two relevant results out of four gives 0.5 precision.
    """

    retrieved = [
        "chunk-a",
        "chunk-x",
        "chunk-c",
        "chunk-y",
    ]

    relevant = {
        "chunk-a",
        "chunk-c",
    }

    assert (
        precision_at_k(
            retrieved,
            relevant,
            k=4,
        )
        == 0.5
    )


def test_reciprocal_rank():
    """
    First relevant result at rank 2 gives reciprocal rank 0.5.
    """

    retrieved = [
        "chunk-x",
        "chunk-a",
        "chunk-c",
    ]

    relevant = {
        "chunk-a",
        "chunk-c",
    }

    assert (
        reciprocal_rank(
            retrieved,
            relevant,
        )
        == 0.5
    )


def test_reciprocal_rank_when_not_found():
    """
    No relevant result should produce reciprocal rank 0.
    """

    retrieved = [
        "chunk-x",
        "chunk-y",
    ]

    relevant = {
        "chunk-a",
    }

    assert (
        reciprocal_rank(
            retrieved,
            relevant,
        )
        == 0.0
    )


def test_mean_reciprocal_rank():
    """
    MRR should average reciprocal ranks across queries.
    """

    ranked_lists = [
        [
            "chunk-a",
            "chunk-x",
        ],
        [
            "chunk-x",
            "chunk-b",
        ],
    ]

    relevant_sets = [
        {"chunk-a"},
        {"chunk-b"},
    ]

    # Query 1: RR = 1.0
    # Query 2: RR = 0.5
    # MRR = 0.75
    assert (
        mean_reciprocal_rank(
            ranked_lists,
            relevant_sets,
        )
        == 0.75
    )


def test_ndcg_perfect_ranking():
    """
    If all relevant results occur first, nDCG should be 1.
    """

    retrieved = [
        "chunk-a",
        "chunk-b",
        "chunk-x",
    ]

    relevant = {
        "chunk-a",
        "chunk-b",
    }

    assert ndcg_at_k(
        retrieved,
        relevant,
        k=3,
    ) == pytest.approx(1.0)


def test_ndcg_rewards_higher_ranking():
    """
    A relevant result appearing earlier should produce a
    better nDCG score.
    """

    relevant = {
        "chunk-a",
    }

    good = [
        "chunk-a",
        "chunk-x",
        "chunk-y",
    ]

    bad = [
        "chunk-x",
        "chunk-y",
        "chunk-a",
    ]

    assert ndcg_at_k(
        good,
        relevant,
        k=3,
    ) > ndcg_at_k(
        bad,
        relevant,
        k=3,
    )


def test_invalid_k_is_rejected():
    """
    Metric cutoffs must be positive.
    """

    with pytest.raises(ValueError):
        recall_at_k(
            ["chunk-a"],
            {"chunk-a"},
            k=0,
        )

    with pytest.raises(ValueError):
        precision_at_k(
            ["chunk-a"],
            {"chunk-a"},
            k=0,
        )

    with pytest.raises(ValueError):
        ndcg_at_k(
            ["chunk-a"],
            {"chunk-a"},
            k=0,
        )
