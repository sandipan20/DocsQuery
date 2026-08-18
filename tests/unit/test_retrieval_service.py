"""
Unit tests for the retrieval service.
"""

from unittest.mock import MagicMock

import pytest

from app.retrieval.models import RetrievalResult
from app.services.retrieval_service import RetrievalService


def create_result() -> RetrievalResult:
    """
    Create a predictable retrieval result for testing.
    """

    return RetrievalResult(
        chunk_id="chunk-001",
        document_id="doc-001",
        text="Test retrieval result.",
        source="test.pdf",
        page_number=1,
        chunk_index=0,
        score=0.01,
    )


def test_search_delegates_to_retriever():
    """
    The service should delegate the search operation to the
    configured hybrid retriever.
    """

    retriever = MagicMock()

    retriever.retrieve.return_value = [create_result()]

    service = RetrievalService(
        retriever=retriever,
    )

    results = service.search(
        query="Python",
        limit=5,
    )

    retriever.retrieve.assert_called_once_with(
        query="Python",
        limit=5,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-001"


def test_retriever_errors_are_propagated():
    """
    The service should not silently hide retrieval errors.

    This is important because infrastructure or validation
    failures should be visible to the API layer.
    """

    retriever = MagicMock()

    retriever.retrieve.side_effect = ValueError("Query cannot be empty.")

    service = RetrievalService(
        retriever=retriever,
    )

    with pytest.raises(ValueError):
        service.search(
            query="",
            limit=5,
        )
