"""
Tests for negative-query retrieval metrics.
"""

import pytest

from app.evaluation.metrics import (
    false_retrieval_rate_at_k,
    mean_false_retrieval_rate_at_k,
)


def test_false_retrieval_rate_returns_one_when_results_exist() -> None:
    retrieved_ids = ["chunk-1", "chunk-2", "chunk-3"]

    assert (
        false_retrieval_rate_at_k(
            retrieved_ids,
            1,
        )
        == 1.0
    )


def test_false_retrieval_rate_returns_zero_when_no_results_exist() -> None:
    retrieved_ids: list[str] = []

    assert (
        false_retrieval_rate_at_k(
            retrieved_ids,
            5,
        )
        == 0.0
    )


def test_false_retrieval_rate_returns_one_when_result_exists_within_k() -> None:
    retrieved_ids = ["chunk-1", "chunk-2"]

    assert (
        false_retrieval_rate_at_k(
            retrieved_ids,
            2,
        )
        == 1.0
    )


def test_false_retrieval_rate_returns_one_for_results_beyond_k() -> None:
    retrieved_ids = ["chunk-1", "chunk-2", "chunk-3"]

    assert (
        false_retrieval_rate_at_k(
            retrieved_ids,
            2,
        )
        == 1.0
    )


def test_false_retrieval_rate_validates_k() -> None:
    with pytest.raises(ValueError):
        false_retrieval_rate_at_k(
            ["chunk-1"],
            0,
        )


def test_mean_false_retrieval_rate_returns_zero_for_empty_dataset() -> None:
    assert (
        mean_false_retrieval_rate_at_k(
            [],
            5,
        )
        == 0.0
    )


def test_mean_false_retrieval_rate_all_queries_retrieve_results() -> None:
    ranked_lists = [
        ["chunk-1"],
        ["chunk-2", "chunk-3"],
        ["chunk-4"],
    ]

    assert (
        mean_false_retrieval_rate_at_k(
            ranked_lists,
            5,
        )
        == 1.0
    )


def test_mean_false_retrieval_rate_all_queries_return_nothing() -> None:
    ranked_lists = [
        [],
        [],
        [],
    ]

    assert (
        mean_false_retrieval_rate_at_k(
            ranked_lists,
            5,
        )
        == 0.0
    )


def test_mean_false_retrieval_rate_mixed_results() -> None:
    ranked_lists = [
        ["chunk-1"],
        [],
        ["chunk-2"],
        [],
    ]

    assert (
        mean_false_retrieval_rate_at_k(
            ranked_lists,
            5,
        )
        == 0.5
    )


def test_mean_false_retrieval_rate_respects_k() -> None:
    ranked_lists = [
        [],
        ["chunk-1", "chunk-2"],
        [],
    ]

    assert mean_false_retrieval_rate_at_k(
        ranked_lists,
        1,
    ) == pytest.approx(1 / 3)


def test_mean_false_retrieval_rate_validates_k() -> None:
    with pytest.raises(ValueError):
        mean_false_retrieval_rate_at_k(
            [["chunk-1"]],
            0,
        )
