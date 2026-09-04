"""
Tests for retrieval baseline storage.
"""

from pathlib import Path

import pytest

from app.evaluation.baseline import (
    compare_results,
    load_baseline,
    save_baseline,
)
from app.evaluation.results import (
    EvaluationResult,
    MetricResult,
)


def create_result() -> EvaluationResult:
    """
    Create a predictable benchmark result.
    """

    return EvaluationResult(
        strategy="BM25",
        num_examples=10,
        metrics=MetricResult(
            recall_at_1=0.5,
            recall_at_3=0.7,
            recall_at_5=0.8,
            precision_at_1=0.5,
            precision_at_3=0.4,
            precision_at_5=0.3,
            mrr=0.65,
            ndcg_at_5=0.7,
        ),
    )


def test_save_and_load_baseline(
    tmp_path: Path,
):
    """
    A saved baseline should load identically.
    """

    file_path = tmp_path / "baseline.json"

    results = [create_result()]

    save_baseline(
        dataset_version="1.1.0",
        results=results,
        file_path=str(file_path),
    )

    baseline = load_baseline(str(file_path))

    assert baseline.dataset_version == "1.1.0"
    assert baseline.results == results


def test_missing_baseline_fails(
    tmp_path: Path,
):
    """
    Loading a missing baseline should fail clearly.
    """

    file_path = tmp_path / "missing.json"

    try:
        load_baseline(str(file_path))

    except FileNotFoundError as exc:
        assert "Baseline not found" in str(exc)

    else:
        raise AssertionError("Expected FileNotFoundError")


def test_baseline_contains_expected_strategies(
    tmp_path: Path,
):
    """
    A retrieval baseline should contain all four retrieval
    strategies.
    """

    file_path = tmp_path / "baseline.json"

    strategies = {
        "BM25",
        "Vector",
        "Hybrid RRF",
        "Hybrid RRF + Reranker",
    }

    results = []

    for strategy in strategies:
        results.append(
            EvaluationResult(
                strategy=strategy,
                num_examples=10,
                metrics=MetricResult(
                    recall_at_1=0.5,
                    recall_at_3=0.6,
                    recall_at_5=0.7,
                    precision_at_1=0.5,
                    precision_at_3=0.4,
                    precision_at_5=0.3,
                    mrr=0.6,
                    ndcg_at_5=0.6,
                ),
            )
        )

    save_baseline(
        dataset_version="1.1.0",
        results=results,
        file_path=str(file_path),
    )

    baseline = load_baseline(str(file_path))

    actual = {result.strategy for result in baseline.results}

    assert actual == strategies


def test_compare_results_calculates_metric_deltas():
    """
    Current results should be compared against baseline
    metric-by-metric.
    """

    baseline = [create_result()]

    current = [
        EvaluationResult(
            strategy="BM25",
            num_examples=10,
            metrics=MetricResult(
                recall_at_1=0.6,
                recall_at_3=0.8,
                recall_at_5=0.9,
                precision_at_1=0.6,
                precision_at_3=0.5,
                precision_at_5=0.4,
                mrr=0.75,
                ndcg_at_5=0.8,
            ),
        )
    ]

    comparison = compare_results(
        current=current,
        baseline=baseline,
    )

    assert comparison["BM25"]["mrr"] == pytest.approx(0.10)

    assert comparison["BM25"]["recall_at_5"] == pytest.approx(0.10)
