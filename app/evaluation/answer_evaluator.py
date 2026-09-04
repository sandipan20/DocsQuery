"""
DocsQuery - Answer Evaluator

Coordinates deterministic citation checks and semantic
groundedness evaluation.

This evaluator belongs to the evaluation layer and is not
part of the normal production query path.
"""

from app.evaluation.answer_metrics import (
    AnswerMetrics,
)
from app.evaluation.answer_models import (
    AnswerEvaluationResult,
)
from app.evaluation.groundedness import (
    GroundednessEvaluator,
)
from app.generation.context_builder import (
    CitationContext,
)


class AnswerEvaluator:
    """
    Evaluates generated RAG answers.
    """

    def __init__(
        self,
        answer_metrics: AnswerMetrics | None = None,
        groundedness_evaluator: (GroundednessEvaluator | None) = None,
    ):
        """
        Initialize answer evaluation dependencies.

        Args:
            answer_metrics:
                Deterministic citation-based metrics.

            groundedness_evaluator:
                Optional NLI-based groundedness evaluator.

                It is optional because loading an NLI model is
                relatively expensive and should not be required
                for deterministic tests.
        """

        self.answer_metrics = answer_metrics or AnswerMetrics()

        self.groundedness_evaluator = groundedness_evaluator

    def evaluate(
        self,
        example_id: str,
        answer: str,
        contexts: list[CitationContext],
    ) -> AnswerEvaluationResult:
        """
        Evaluate one generated answer.

        Args:
            example_id:
                Evaluation example identifier.

            answer:
                Generated answer.

            contexts:
                Retrieved evidence.

        Returns:
            Structured answer evaluation result.
        """

        # ----------------------------------------------------
        # Deterministic citation validation
        # ----------------------------------------------------

        citation_valid = self.answer_metrics.citation_validity(
            answer,
            contexts,
        )

        # ----------------------------------------------------
        # Citation coverage
        # ----------------------------------------------------

        citation_coverage = self.answer_metrics.citation_coverage(answer)

        # ----------------------------------------------------
        # Groundedness
        # ----------------------------------------------------
        #
        # Default to zero so the value is always defined,
        # even when the expensive NLI evaluator is disabled.

        groundedness_score = 0.0
        grounded = False

        if self.groundedness_evaluator is not None and contexts:
            # Combine all retrieved evidence into the premise
            # used by the NLI evaluator.
            evidence = "\n\n".join(context.result.text for context in contexts)

            groundedness_score = self.groundedness_evaluator.evaluate(
                answer=answer,
                evidence=evidence,
            )

            # This boolean is only a convenient classification.
            # The continuous score is preserved because it contains
            # more information than True/False.
            grounded = groundedness_score >= 0.5

        return AnswerEvaluationResult(
            example_id=example_id,
            citation_valid=citation_valid,
            citation_coverage=citation_coverage,
            grounded=grounded,
            groundedness_score=groundedness_score,
        )
