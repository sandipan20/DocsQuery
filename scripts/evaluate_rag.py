"""
DocsQuery - End-to-End RAG Evaluation CLI

Evaluates the complete DocsQuery pipeline.

Pipeline:

    Evaluation Question
          ↓
    Hybrid Retrieval
          ↓
    Cross-Encoder
          ↓
    Gemini
          ↓
    Citation Validation
          ↓
    Groundedness
          ↓
    Correctness
"""

import json
from pathlib import Path

from app.container import AppContainer
from app.evaluation.answer_correctness import (
    AnswerCorrectnessEvaluator,
)
from app.evaluation.answer_evaluator import (
    AnswerEvaluator,
)
from app.evaluation.dataset import (
    load_evaluation_dataset,
)
from app.evaluation.e2e_evaluator import (
    EndToEndEvaluator,
)
from app.evaluation.e2e_results import (
    EndToEndExampleResult,
    EndToEndSummary,
)
from app.evaluation.groundedness import (
    GroundednessEvaluator,
)
from app.retrieval.embeddings import (
    EmbeddingService,
)


def save_results(
    summary: EndToEndSummary,
    results: list[EndToEndExampleResult],
    output_path: str,
) -> None:
    """
    Save complete RAG evaluation results to JSON.
    """

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "summary": summary.model_dump(),
        "examples": [result.model_dump() for result in results],
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def print_summary(
    summary: EndToEndSummary,
) -> None:
    """
    Print aggregate RAG evaluation metrics.
    """

    print()
    print("=" * 70)
    print("DocsQuery End-to-End RAG Evaluation")
    print("=" * 70)

    print(f"Dataset version: {summary.dataset_version}")

    print(f"Examples:         {summary.num_examples}")

    print(f"Retrieval success: {summary.retrieval_success_rate:.4f}")

    print(f"Citation validity: {summary.citation_validity_rate:.4f}")

    print(f"Citation coverage: {summary.average_citation_coverage:.4f}")

    print(f"Groundedness:      {summary.average_groundedness:.4f}")

    print(f"Grounded answers:  {summary.grounded_answer_rate:.4f}")

    if summary.average_correctness is not None:
        print(f"Correctness:       {summary.average_correctness:.4f}")

    print("=" * 70)


def main() -> None:
    """
    Run end-to-end RAG evaluation.
    """

    # --------------------------------------------------------
    # Load evaluation dataset.
    # --------------------------------------------------------

    dataset = load_evaluation_dataset("data/evaluation/retrieval_dataset.json")

    print()
    print(f"Evaluating {len(dataset.examples)} examples...")

    # --------------------------------------------------------
    # Create the production RAG container.
    # --------------------------------------------------------

    container = AppContainer()

    # Restore the BM25 corpus.
    container.load_indexes()

    # --------------------------------------------------------
    # Create evaluation-only models.
    # --------------------------------------------------------

    groundedness_evaluator = GroundednessEvaluator()

    correctness_evaluator = AnswerCorrectnessEvaluator(
        embedding_service=EmbeddingService()
    )

    # --------------------------------------------------------
    # Create the end-to-end evaluator.
    # --------------------------------------------------------

    evaluator = EndToEndEvaluator(
        rag_service=container.rag_service,
        answer_evaluator=AnswerEvaluator(groundedness_evaluator=groundedness_evaluator),
        correctness_evaluator=correctness_evaluator,
    )

    # --------------------------------------------------------
    # Evaluate.
    # --------------------------------------------------------

    summary, results = evaluator.evaluate(
        examples=dataset.examples,
        dataset_version=dataset.version,
    )

    # --------------------------------------------------------
    # Print aggregate metrics.
    # --------------------------------------------------------

    print_summary(summary)

    # --------------------------------------------------------
    # Save machine-readable results.
    # --------------------------------------------------------

    output_path = "data/evaluation/rag_results.json"

    save_results(
        summary=summary,
        results=results,
        output_path=output_path,
    )

    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
