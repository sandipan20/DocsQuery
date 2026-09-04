"""
DocsQuery - Answer Evaluation Metrics

Deterministic metrics for generated answers.

These metrics intentionally avoid an LLM judge.

They verify properties such as:

    citation validity
    citation coverage
    presence of evidence
"""

from app.generation.citation_validator import (
    CitationValidator,
)
from app.generation.context_builder import CitationContext


class AnswerMetrics:
    """
    Calculates deterministic answer-quality metrics.
    """

    def __init__(
        self,
        citation_validator: CitationValidator | None = None,
    ):
        """
        Initialize the metric calculator.
        """

        self.citation_validator = citation_validator or CitationValidator()

    def citation_validity(
        self,
        answer: str,
        contexts: list[CitationContext],
    ) -> bool:
        """
        Determine whether all citations in an answer are valid.

        Returns:
            True if citation validation succeeds.
        """

        try:
            self.citation_validator.validate(
                answer=answer,
                contexts=contexts,
            )

        except Exception:
            return False

        return True

    def citation_coverage(
        self,
        answer: str,
    ) -> float:
        """
        Calculate citation coverage.
        """

        return self.citation_validator.citation_coverage(answer)

    def has_evidence(
        self,
        contexts: list[CitationContext],
    ) -> bool:
        """
        Check whether evidence exists.

        This is intentionally simple. It allows the evaluation
        pipeline to distinguish:

            no evidence
            from
            evidence available
        """

        return bool(contexts)
