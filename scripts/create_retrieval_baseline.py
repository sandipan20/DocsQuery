"""
DocsQuery - Create Retrieval Baseline

Runs the current retrieval benchmark and stores its results
as the official baseline.

IMPORTANT:
Only run this when the current retrieval implementation and
evaluation dataset are considered a valid checkpoint.

Example:

    python -m scripts.create_retrieval_baseline
"""

from app.evaluation.baseline import save_baseline
from app.evaluation.dataset import (
    load_evaluation_dataset,
)
from app.evaluation.evaluator import (
    RetrievalEvaluator,
)
from scripts.evaluate_retrieval import (
    build_strategies,
)


def main() -> None:
    """
    Run the benchmark and save the results as a baseline.
    """

    dataset = load_evaluation_dataset("data/evaluation/retrieval_dataset.json")

    strategies = build_strategies()

    evaluator = RetrievalEvaluator()

    results = []

    print()
    print("DocsQuery Retrieval Baseline")
    print("=" * 70)
    print(f"Dataset version: {dataset.version}")
    print(f"Examples: {len(dataset.examples)}")

    for strategy_name, retriever in strategies.items():
        print()
        print(f"Evaluating {strategy_name}...")

        result = evaluator.evaluate(
            strategy=strategy_name,
            examples=dataset.examples,
            retriever=retriever,
        )

        results.append(result)

    output_path = "data/evaluation/retrieval_baseline.json"

    save_baseline(
        dataset_version=dataset.version,
        results=results,
        file_path=output_path,
    )

    print()
    print(f"Baseline saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
