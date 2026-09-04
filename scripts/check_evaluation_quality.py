"""
DocsQuery - Evaluation Quality Gate CLI

Runs retrieval and answer evaluation, compares retrieval
metrics against the accepted baseline, and exits with status 1
when quality requirements are violated.

This non-zero exit status is what GitHub Actions will use later
to fail a CI job.
"""

import json
import sys
from pathlib import Path

from app.evaluation.baseline import (
    load_baseline,
)
from app.evaluation.dataset import (
    load_evaluation_dataset,
)
from app.evaluation.e2e_results import (
    EndToEndSummary,
)
from app.evaluation.evaluator import (
    RetrievalEvaluator,
)
from app.evaluation.quality_gate import (
    EvaluationQualityGate,
    QualityGateConfig,
)
from scripts.evaluate_retrieval import (
    build_strategies,
)


def load_retrieval_results():
    """
    Run the current retrieval benchmark.

    Returns:
        Current retrieval evaluation results.
    """

    dataset = load_evaluation_dataset("data/evaluation/retrieval_dataset.json")

    strategies = build_strategies()

    evaluator = RetrievalEvaluator()

    results = []

    for strategy_name, retriever in strategies.items():
        results.append(
            evaluator.evaluate(
                strategy=strategy_name,
                examples=dataset.examples,
                retriever=retriever,
            )
        )

    return results, dataset.version


def load_rag_results():
    """
    Load previously generated RAG evaluation results.

    We deliberately don't call Gemini twice during the same
    quality-gate execution.
    """

    path = Path("data/evaluation/rag_results.json")

    if not path.exists():
        raise FileNotFoundError(
            "Run `python -m scripts.evaluate_rag` before checking answer quality."
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    return EndToEndSummary.model_validate(data["summary"])


def print_failures(
    failures,
) -> None:
    """
    Print human-readable quality-gate failures.
    """

    print()
    print("QUALITY GATE FAILURES")
    print("=" * 70)

    for failure in failures:
        print(f"[FAIL] {failure.metric}: {failure.message}")


def main() -> None:
    """
    Execute the complete quality gate.
    """

    # --------------------------------------------------------
    # Load baseline.
    # --------------------------------------------------------

    baseline = load_baseline("data/evaluation/retrieval_baseline.json")

    # --------------------------------------------------------
    # Run retrieval evaluation.
    # --------------------------------------------------------

    retrieval_results, dataset_version = load_retrieval_results()

    if dataset_version != baseline.dataset_version:
        print(
            "ERROR: Evaluation dataset version does not match the retrieval baseline."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Load previously generated RAG answer metrics.
    # --------------------------------------------------------

    answer_summary = load_rag_results()

    # --------------------------------------------------------
    # Load quality-gate configuration.
    # --------------------------------------------------------

    config = QualityGateConfig("configs/evaluation.yaml")

    gate = EvaluationQualityGate(config)

    # --------------------------------------------------------
    # Execute all checks.
    # --------------------------------------------------------

    result = gate.check(
        current_retrieval=retrieval_results,
        baseline=baseline,
        answer_summary=answer_summary,
    )

    # --------------------------------------------------------
    # Report.
    # --------------------------------------------------------

    print()
    print("DocsQuery Evaluation Quality Gate")
    print("=" * 70)

    print(f"Dataset version: {dataset_version}")

    print(f"Baseline version: {baseline.dataset_version}")

    if result.passed:
        print()
        print("QUALITY GATE: PASS")
        print("=" * 70)

        return

    print_failures(result.failures)

    print()
    print("QUALITY GATE: FAIL")
    print("=" * 70)

    # Non-zero exit code is essential for CI.
    sys.exit(1)


if __name__ == "__main__":
    main()
