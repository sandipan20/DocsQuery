"""
DocsQuery - Benchmark Summary

Reads the saved retrieval benchmark results and prints a
compact comparison table.

This script does not run retrieval itself.

It only reads:

    data/evaluation/retrieval_results.json

and presents the results in a readable form.
"""

import json
from pathlib import Path

RESULTS_PATH = Path("data/evaluation/retrieval_results.json")


def main() -> None:
    """
    Load benchmark results and print a compact summary.
    """

    # --------------------------------------------------------
    # Make sure the benchmark has already been executed.
    # --------------------------------------------------------

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Benchmark results were not found at "
            f"{RESULTS_PATH}.\n"
            "Run this first:\n"
            "python scripts/evaluate_retrieval.py"
        )

    # --------------------------------------------------------
    # Read the generated JSON results.
    # --------------------------------------------------------

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    if not results:
        print("No benchmark results found.")
        return

    # --------------------------------------------------------
    # Print the comparison table.
    # --------------------------------------------------------

    print()
    print("DocsQuery Retrieval Benchmark Summary")
    print("=" * 90)

    print(
        f"{'Strategy':30s}"
        f"{'Recall@5':>12s}"
        f"{'Precision@5':>15s}"
        f"{'MRR':>10s}"
        f"{'nDCG@5':>12s}"
    )

    print("-" * 90)

    for result in results:
        metrics = result["metrics"]

        print(
            f"{result['strategy']:30s}"
            f"{metrics['recall_at_5']:12.4f}"
            f"{metrics['precision_at_5']:15.4f}"
            f"{metrics['mrr']:10.4f}"
            f"{metrics['ndcg_at_5']:12.4f}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()
