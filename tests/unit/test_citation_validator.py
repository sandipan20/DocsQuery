"""
Unit tests for citation validation.
"""

import pytest

from app.generation.citation_validator import (
    CitationValidationError,
    CitationValidator,
)
from app.generation.context_builder import (
    CitationContext,
)
from app.retrieval.models import RetrievalResult


def create_context(
    citation_id: str,
    text: str,
) -> CitationContext:
    """
    Create a predictable citation context.
    """

    result = RetrievalResult(
        chunk_id=f"{citation_id}-chunk",
        document_id="doc-001",
        text=text,
        source="test.pdf",
        page_number=1,
        chunk_index=0,
        score=0.9,
    )

    return CitationContext(
        citation_id=citation_id,
        result=result,
    )


@pytest.fixture
def contexts():
    """
    Provide predictable evidence.
    """

    return [
        create_context(
            "C1",
            "Python is a programming language.",
        ),
        create_context(
            "C2",
            "Python uses indentation.",
        ),
    ]


@pytest.fixture
def validator():
    """
    Create a citation validator.
    """

    return CitationValidator()


def test_valid_citations_pass(
    validator,
    contexts,
):
    """
    Valid citation IDs should pass.
    """

    answer = "Python is a programming language. [C1] Python uses indentation. [C2]"

    validator.validate(
        answer,
        contexts,
    )


def test_unknown_citation_fails(
    validator,
    contexts,
):
    """
    Unknown citation IDs should be rejected.
    """

    answer = "Python is a programming language. [C7]"

    with pytest.raises(
        CitationValidationError,
        match="unknown citation",
    ):
        validator.validate(
            answer,
            contexts,
        )


def test_missing_citation_fails(
    validator,
    contexts,
):
    """
    A factual sentence without a citation should fail.
    """

    answer = "Python is a programming language."

    with pytest.raises(
        CitationValidationError,
        match="missing a citation",
    ):
        validator.validate(
            answer,
            contexts,
        )


def test_empty_answer_fails(
    validator,
    contexts,
):
    """
    Empty answers should be rejected.
    """

    with pytest.raises(
        CitationValidationError,
        match="Answer is empty",
    ):
        validator.validate(
            "",
            contexts,
        )


def test_empty_context_fails(
    validator,
):
    """
    Answers cannot be validated without evidence.
    """

    with pytest.raises(
        CitationValidationError,
        match="No citation context",
    ):
        validator.validate(
            "Python is a language. [C1]",
            [],
        )


def test_multiple_citations_are_allowed(
    validator,
    contexts,
):
    """
    A sentence can cite multiple evidence chunks.
    """

    answer = "Python is a programming language and uses indentation. [C1] [C2]"

    validator.validate(
        answer,
        contexts,
    )


def test_extract_citations_preserves_order(
    validator,
):
    """
    Citation extraction should preserve first appearance
    order and remove duplicates.
    """

    answer = "Python is a language. [C2] It is widely used. [C1] It is readable. [C2]"

    citations = validator.extract_citations(answer)

    assert citations == [
        "C2",
        "C1",
    ]
