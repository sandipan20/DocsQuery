"""
Unit tests for the generation service.
"""

from unittest.mock import MagicMock

import pytest

from app.generation.citation_validator import (
    CitationValidator,
)
from app.generation.context_builder import (
    ContextBuilder,
)
from app.generation.models import GeneratedAnswer
from app.retrieval.models import RetrievalResult
from app.services.generation_service import (
    GenerationService,
)


def create_result() -> RetrievalResult:
    """
    Create a predictable retrieval result.
    """

    return RetrievalResult(
        chunk_id="chunk-001",
        document_id="doc-001",
        text=("Python is a programming language."),
        source="test.pdf",
        page_number=1,
        chunk_index=0,
        score=0.9,
    )


def create_service(
    answer: str,
) -> GenerationService:
    """
    Create a generation service using a fake LLM.
    """

    llm = MagicMock()

    llm.generate.return_value = answer

    return GenerationService(
        llm_service=llm,
        context_builder=ContextBuilder(),
        citation_validator=(CitationValidator()),
    )


def test_generate_returns_validated_answer():
    """
    A correctly cited answer should be returned.
    """

    service = create_service("Python is a programming language. [C1]")

    result = service.generate(
        query="What is Python?",
        results=[create_result()],
    )

    assert isinstance(
        result,
        GeneratedAnswer,
    )

    assert result.answer == "Python is a programming language. [C1]"

    assert result.citations == ["C1"]


def test_invalid_citation_is_rejected():
    """
    Answers containing nonexistent citations must fail.
    """

    service = create_service("Python is a programming language. [C99]")

    with pytest.raises(
        Exception,
        match="unknown citation",
    ):
        service.generate(
            query="What is Python?",
            results=[create_result()],
        )


def test_empty_results_are_rejected():
    """
    The LLM should never be called without evidence.
    """

    service = create_service("Python is a programming language. [C1]")

    with pytest.raises(
        ValueError,
        match="without.*evidence",
    ):
        service.generate(
            query="What is Python?",
            results=[],
        )


def test_empty_query_is_rejected():
    """
    Empty user queries should be rejected.
    """

    service = create_service("Python is a programming language. [C1]")

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        service.generate(
            query="",
            results=[create_result()],
        )
