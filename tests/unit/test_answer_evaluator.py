"""
Tests for the answer evaluation coordinator.
"""

from app.evaluation.answer_evaluator import (
    AnswerEvaluator,
)
from app.generation.context_builder import (
    CitationContext,
)
from app.retrieval.models import RetrievalResult


class FakeGroundednessEvaluator:
    """
    Fake groundedness evaluator used for unit tests.

    It returns a predictable score without loading the real
    NLI model.
    """

    def __init__(self, score: float):
        self.score = score

    def evaluate(
        self,
        answer: str,
        evidence: str,
    ) -> float:
        """
        Return the predefined groundedness score.
        """

        return self.score


def create_context() -> CitationContext:
    """
    Create predictable test evidence.
    """

    result = RetrievalResult(
        chunk_id="chunk-001",
        document_id="doc-001",
        text="Python is a programming language.",
        source="test.pdf",
        page_number=1,
        chunk_index=0,
        score=0.9,
    )

    return CitationContext(
        citation_id="C1",
        result=result,
    )


def test_valid_answer_is_evaluated():
    """
    A valid cited answer should produce valid evaluation metrics.
    """

    evaluator = AnswerEvaluator(groundedness_evaluator=(FakeGroundednessEvaluator(0.9)))

    result = evaluator.evaluate(
        example_id="q001",
        answer=("Python is a programming language. [C1]"),
        contexts=[create_context()],
    )

    assert result.citation_valid is True
    assert result.citation_coverage == 1.0
    assert result.groundedness_score == 0.9
    assert result.grounded is True


def test_low_groundedness_is_flagged():
    """
    A low groundedness score should be classified as not grounded.
    """

    evaluator = AnswerEvaluator(groundedness_evaluator=(FakeGroundednessEvaluator(0.2)))

    result = evaluator.evaluate(
        example_id="q001",
        answer=("Python is a programming language. [C1]"),
        contexts=[create_context()],
    )

    assert result.grounded is False
    assert result.groundedness_score == 0.2


def test_invalid_citation_is_detected():
    """
    Unknown citation IDs should be detected as invalid.
    """

    evaluator = AnswerEvaluator()

    result = evaluator.evaluate(
        example_id="q001",
        answer=("Python is a programming language. [C99]"),
        contexts=[create_context()],
    )

    assert result.citation_valid is False


def test_no_groundedness_evaluator():
    """
    Deterministic evaluation should still work when the
    expensive NLI model is not configured.
    """

    evaluator = AnswerEvaluator()

    result = evaluator.evaluate(
        example_id="q001",
        answer=("Python is a programming language. [C1]"),
        contexts=[create_context()],
    )

    assert result.citation_valid is True
    assert result.citation_coverage == 1.0
    assert result.grounded is False
    assert result.groundedness_score == 0.0
