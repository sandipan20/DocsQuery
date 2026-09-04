"""
DocsQuery - RAG Evaluation Summary

Prints a compact summary of end-to-end RAG evaluation results.
"""

import json
from pathlib import Path


def main() -> None:
    """
    Print end-to-end RAG metrics.
    """

    path = Path("data/evaluation/rag_results.json")

    if not path.exists():
        raise FileNotFoundError(f"RAG evaluation results not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    summary = data["summary"]

    print()
    print("DocsQuery RAG Evaluation Summary")
    print("=" * 60)

    print(f"Examples:            {summary['num_examples']}")

    print(f"Retrieval success:   {summary['retrieval_success_rate']:.4f}")

    print(f"Citation validity:   {summary['citation_validity_rate']:.4f}")

    print(f"Citation coverage:   {summary['average_citation_coverage']:.4f}")

    print(f"Groundedness:         {summary['average_groundedness']:.4f}")

    print(f"Grounded answers:     {summary['grounded_answer_rate']:.4f}")

    if summary["average_correctness"] is not None:
        print(f"Correctness:          {summary['average_correctness']:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
