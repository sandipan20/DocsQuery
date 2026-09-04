"""
DocsQuery - End-to-End RAG Evaluator

Evaluates the complete RAG pipeline:

    Query
      ↓
    Retrieval
      ↓
    Reranking
      ↓
    Generation
      ↓
    Citation validation
      ↓
    Groundedness / correctness metrics
"""

from app.evaluation.answer_correctness import (
    AnswerCorrectnessEvaluator,
)
from app.evaluation.answer_evaluator import (
    AnswerEvaluator,
)
from app.evaluation.context import (
    build_evaluation_contexts,
)
from app.evaluation.e2e_results import (
    EndToEndExampleResult,
    EndToEndSummary,
)
from app.evaluation.models import EvaluationExample
from app.services.rag_service import (
    InsufficientEvidenceError,
    RAGService,
)


class EndToEndEvaluator:
    """
    Evaluates the complete RAG system.
    """

    def __init__(
        self,
        rag_service: RAGService,
        answer_evaluator: AnswerEvaluator,
        correctness_evaluator: (AnswerCorrectnessEvaluator | None) = None,
    ):
        """
        Initialize the evaluator.

        Args:
            rag_service:
                Production RAG pipeline.

            answer_evaluator:
                Citation and groundedness evaluator.

            correctness_evaluator:
                Optional semantic correctness evaluator.
        """

        self.rag_service = rag_service
        self.answer_evaluator = answer_evaluator
        self.correctness_evaluator = correctness_evaluator

    def evaluate_example(
        self,
        example: EvaluationExample,
    ) -> EndToEndExampleResult:
        """
        Evaluate one question.

        Args:
            example:
                Ground-truth evaluation example.

        Returns:
            Structured result.
        """

        try:
            response = self.rag_service.query(
                query=example.query,
                top_k=5,
            )

        except InsufficientEvidenceError:
            return EndToEndExampleResult(
                example_id=example.example_id,
                query=example.query,
                retrieved=False,
                citation_valid=False,
                citation_coverage=0.0,
                groundedness_score=0.0,
                grounded=False,
                correctness_score=None,
                answer=None,
            )

        answer = response.generated_answer.answer

        # Build citation-aware evidence using the exact same
        # ordering used by the generation layer.
        contexts = build_evaluation_contexts(response.results)

        answer_metrics = self.answer_evaluator.evaluate(
            example_id=example.example_id,
            answer=answer,
            contexts=contexts,
        )

        correctness_score = None

        if self.correctness_evaluator is not None and example.reference_answer:
            correctness_score = self.correctness_evaluator.evaluate(
                answer=answer,
                reference_answer=(example.reference_answer),
            )

        return EndToEndExampleResult(
            example_id=example.example_id,
            query=example.query,
            retrieved=True,
            citation_valid=(answer_metrics.citation_valid),
            citation_coverage=(answer_metrics.citation_coverage),
            groundedness_score=(answer_metrics.groundedness_score),
            grounded=(answer_metrics.grounded),
            correctness_score=correctness_score,
            answer=answer,
        )

    def evaluate(
        self,
        examples: list[EvaluationExample],
        dataset_version: str,
    ) -> tuple[
        EndToEndSummary,
        list[EndToEndExampleResult],
    ]:
        """
        Evaluate the complete dataset.

        Returns:
            Aggregate summary and per-example results.
        """

        if not examples:
            raise ValueError("Cannot evaluate an empty dataset.")

        results = [self.evaluate_example(example) for example in examples]

        retrieved_count = sum(result.retrieved for result in results)

        citation_valid_count = sum(result.citation_valid for result in results)

        citation_coverage = sum(result.citation_coverage for result in results) / len(
            results
        )

        groundedness = sum(result.groundedness_score for result in results) / len(
            results
        )

        grounded_count = sum(result.grounded for result in results)

        correctness_values = [
            result.correctness_score
            for result in results
            if result.correctness_score is not None
        ]

        average_correctness = (
            sum(correctness_values) / len(correctness_values)
            if correctness_values
            else None
        )

        summary = EndToEndSummary(
            dataset_version=dataset_version,
            num_examples=len(results),
            retrieval_success_rate=(retrieved_count / len(results)),
            citation_validity_rate=(citation_valid_count / len(results)),
            average_citation_coverage=(citation_coverage),
            average_groundedness=(groundedness),
            grounded_answer_rate=(grounded_count / len(results)),
            average_correctness=(average_correctness),
        )

        return summary, results
