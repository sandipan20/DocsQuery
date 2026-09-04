"""
Tests for deterministic answer metrics.
"""

from app.evaluation.answer_metrics import (
    AnswerMetrics,
)
from app.generation.context_builder import (
    CitationContext,
)
from app.retrieval.models import (
    RetrievalResult,
)


def create_context(
    citation_id: str,
) -> CitationContext:
    """
    Create predictable evidence.
    """

    result = RetrievalResult(
        chunk_id=f"{citation_id}-chunk",
        document_id="doc-001",
        text="Python is a programming language.",
        source="test.pdf",
        page_number=1,
        chunk_index=0,
        score=0.9,
    )

    return CitationContext(
        citation_id=citation_id,
        result=result,
    )


def test_valid_citation_returns_true():
    """
    Valid citation IDs should pass.
    """

    metrics = AnswerMetrics()

    contexts = [create_context("C1")]

    answer = "Python is a programming language. [C1]"

    assert (
        metrics.citation_validity(
            answer,
            contexts,
        )
        is True
    )


def test_invalid_citation_returns_false():
    """
    Unknown citation IDs should fail validation.
    """

    metrics = AnswerMetrics()

    contexts = [create_context("C1")]

    answer = "Python is a programming language. [C99]"

    assert (
        metrics.citation_validity(
            answer,
            contexts,
        )
        is False
    )


def test_citation_coverage():
    """
    Citation coverage should match the validator.
    """

    metrics = AnswerMetrics()

    answer = "Python is a language. [C1] It uses indentation."

    assert metrics.citation_coverage(answer) == 0.5


def test_has_evidence():
    """
    Evidence presence should be detected.
    """

    metrics = AnswerMetrics()

    assert metrics.has_evidence([create_context("C1")]) is True

    assert metrics.has_evidence([]) is False
