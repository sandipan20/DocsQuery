"""
DocsQuery - Retrieval Failure Analysis

Runs every evaluation query through all retrieval strategies
and prints detailed per-query results.

This helps answer:

    Which queries fail?
    Which retriever succeeds?
    Where does RRF help?
    Where does reranking hurt?
"""

import json
from pathlib import Path

from app.evaluation.dataset import (
    load_evaluation_dataset,
)
from app.evaluation.diagnostics import (
    QueryDiagnostic,
    analyze_example,
)
from scripts.evaluate_retrieval import (
    build_strategies,
)


def save_diagnostics(
    diagnostics: list[QueryDiagnostic],
    output_path: str,
) -> None:
    """
    Save diagnostic results to JSON.
    """

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            [diagnostic.model_dump() for diagnostic in diagnostics],
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """
    Run detailed retrieval failure analysis.
    """

    # Load the ground-truth dataset.
    dataset = load_evaluation_dataset("data/evaluation/retrieval_dataset.json")

    # Build the same strategies used by the benchmark.
    strategies = build_strategies()

    diagnostics = []

    print()
    print("DocsQuery Retrieval Failure Analysis")
    print("=" * 70)

    for example in dataset.examples:
        strategy_results = {}

        # Run every strategy against the same query.
        for strategy_name, retriever in strategies.items():
            strategy_results[strategy_name] = retriever(
                example.query,
                5,
            )

        diagnostic = analyze_example(
            example_id=example.example_id,
            query=example.query,
            relevant_chunk_ids=(example.relevant_chunk_ids),
            strategy_results=strategy_results,
        )

        diagnostics.append(diagnostic)

        print()
        print(f"{example.example_id}: {example.query}")

        print("Ground truth: " + ", ".join(example.relevant_chunk_ids))

        for strategy in diagnostic.strategies:
            status = "HIT" if strategy.hit else "MISS"

            print(
                f"  {strategy.strategy:25s} {status:4s} {strategy.retrieved_chunk_ids}"
            )

    output_path = "data/evaluation/retrieval_diagnostics.json"

    save_diagnostics(
        diagnostics,
        output_path,
    )

    print()
    print(f"Diagnostics saved to: {output_path}")


if __name__ == "__main__":
    main()
