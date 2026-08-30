"""
Unit tests for the complete RAG service.
"""

from unittest.mock import MagicMock

import pytest

from app.generation.models import GeneratedAnswer
from app.retrieval.models import RetrievalResult
from app.services.rag_service import (
    InsufficientEvidenceError,
    RAGService,
)


def create_result() -> RetrievalResult:
    """
    Create predictable retrieval evidence.
    """

    return RetrievalResult(
        chunk_id="chunk-001",
        document_id="doc-001",
        text="Python is a programming language.",
        source="python.pdf",
        page_number=1,
        chunk_index=0,
        score=0.9,
    )


def create_service(
    results: list[RetrievalResult],
) -> RAGService:
    """
    Create a RAG service with mocked dependencies.
    """

    retrieval_service = MagicMock()

    retrieval_service.search.return_value = results

    generation_service = MagicMock()

    generation_service.generate.return_value = GeneratedAnswer(
        answer=("Python is a programming language. [C1]"),
        citations=["C1"],
    )

    return RAGService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
    )


def test_query_returns_answer():
    """
    A query with evidence should return an answer.
    """

    service = create_service([create_result()])

    response = service.query("What is Python?")

    assert response.query == "What is Python?"

    assert response.generated_answer.answer == "Python is a programming language. [C1]"

    assert len(response.results) == 1


def test_query_passes_top_k_to_retrieval():
    """
    The requested top_k should be passed to retrieval.
    """

    retrieval_service = MagicMock()

    retrieval_service.search.return_value = [create_result()]

    generation_service = MagicMock()

    generation_service.generate.return_value = GeneratedAnswer(
        answer="Python is a language. [C1]",
        citations=["C1"],
    )

    service = RAGService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
    )

    service.query(
        "What is Python?",
        top_k=3,
    )

    retrieval_service.search.assert_called_once_with(
        query="What is Python?",
        limit=3,
    )


def test_no_evidence_is_rejected():
    """
    The LLM should not be called when retrieval returns
    no evidence.
    """

    retrieval_service = MagicMock()

    retrieval_service.search.return_value = []

    generation_service = MagicMock()

    service = RAGService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
    )

    with pytest.raises(
        InsufficientEvidenceError,
    ):
        service.query("Unknown question")

    generation_service.generate.assert_not_called()


def test_empty_query_is_rejected():
    """
    Empty queries should fail before retrieval.
    """

    service = create_service([create_result()])

    with pytest.raises(ValueError):
        service.query("")


def test_invalid_top_k_is_rejected():
    """
    top_k must be positive.
    """

    service = create_service([create_result()])

    with pytest.raises(ValueError):
        service.query(
            "What is Python?",
            top_k=0,
        )
