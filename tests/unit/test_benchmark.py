"""
Tests for retrieval benchmark utilities.
"""

import json
from pathlib import Path

from app.evaluation.results import (
    EvaluationResult,
    MetricResult,
)
from scripts.evaluate_retrieval import save_results


def create_result() -> EvaluationResult:
    """
    Create a predictable benchmark result.
    """

    return EvaluationResult(
        strategy="BM25",
        num_examples=5,
        metrics=MetricResult(
            recall_at_1=1.0,
            recall_at_3=1.0,
            recall_at_5=1.0,
            precision_at_1=1.0,
            precision_at_3=0.5,
            precision_at_5=0.2,
            mrr=1.0,
            ndcg_at_5=1.0,
        ),
    )


def test_save_results(
    tmp_path: Path,
):
    """
    Benchmark results should be serialized to JSON.
    """

    output_path = tmp_path / "results.json"

    save_results(
        results=[create_result()],
        output_path=str(output_path),
    )

    assert output_path.exists()

    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(data) == 1
    assert data[0]["strategy"] == "BM25"
    assert data[0]["num_examples"] == 5
