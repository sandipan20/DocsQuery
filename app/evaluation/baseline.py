"""
DocsQuery - Evaluation Baseline

Defines the persisted baseline used for retrieval regression
testing.

The baseline records the benchmark results from a known-good
version of the application.
"""

from pathlib import Path

from pydantic import BaseModel

from app.evaluation.results import EvaluationResult


class RetrievalBaseline(BaseModel):
    """
    Stored retrieval baseline.

    The baseline contains:
        - dataset version
        - evaluation results
    """

    dataset_version: str

    results: list[EvaluationResult]


def save_baseline(
    dataset_version: str,
    results: list[EvaluationResult],
    file_path: str,
) -> None:
    """
    Save benchmark results as the official baseline.

    Args:
        dataset_version:
            Version of the evaluation dataset used.

        results:
            Benchmark results.

        file_path:
            Destination JSON file.
    """

    baseline = RetrievalBaseline(
        dataset_version=dataset_version,
        results=results,
    )

    path = Path(file_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        baseline.model_dump_json(indent=2),
        encoding="utf-8",
    )


def load_baseline(
    file_path: str,
) -> RetrievalBaseline:
    """
    Load a previously saved baseline.

    Args:
        file_path:
            Path to the baseline JSON.

    Returns:
        Validated RetrievalBaseline.

    Raises:
        FileNotFoundError:
            If the baseline does not exist.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Baseline not found: {file_path}")

    return RetrievalBaseline.model_validate_json(path.read_text(encoding="utf-8"))


def compare_metric(
    current: float,
    baseline: float,
) -> float:
    """
    Calculate the absolute change from baseline.

    Example:

        baseline = 0.80
        current  = 0.75

        result = -0.05
    """

    return current - baseline


def compare_results(
    current: list[EvaluationResult],
    baseline: list[EvaluationResult],
) -> dict[str, dict[str, float]]:
    """
    Compare current benchmark results against the baseline.

    Returns:
        Mapping:

            strategy
                ↓
            metric
                ↓
            delta

    Example:

        {
            "BM25": {
                "mrr": -0.03,
                "ndcg_at_5": 0.01
            }
        }
    """

    baseline_by_strategy = {result.strategy: result for result in baseline}

    comparisons = {}

    for current_result in current:
        baseline_result = baseline_by_strategy.get(current_result.strategy)

        if baseline_result is None:
            continue

        current_metrics = current_result.metrics.model_dump()

        baseline_metrics = baseline_result.metrics.model_dump()

        comparisons[current_result.strategy] = {
            metric: compare_metric(
                current_metrics[metric],
                baseline_metrics[metric],
            )
            for metric in current_metrics
        }

    return comparisons
