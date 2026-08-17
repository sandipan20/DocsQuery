"""
Unit tests for the vector retriever.
"""

from unittest.mock import MagicMock

import pytest

from app.retrieval.models import RetrievalResult
from app.retrieval.vector_retriever import VectorRetriever


def create_result() -> RetrievalResult:
    """
    Create a predictable retrieval result.
    """

    return RetrievalResult(
        chunk_id="chunk-001",
        document_id="doc-001",
        text="Authentication requires a valid token.",
        source="test.pdf",
        page_number=5,
        chunk_index=2,
        score=0.91,
    )


def test_empty_query_is_rejected():
    """
    Empty queries should not be sent to the embedding model.
    """

    embedding_service = MagicMock()
    vector_store = MagicMock()

    retriever = VectorRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_invalid_limit_is_rejected():
    """
    Retrieval limit must be positive.
    """

    embedding_service = MagicMock()
    vector_store = MagicMock()

    retriever = VectorRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            "What is authentication?",
            limit=0,
        )


def test_query_is_embedded():
    """
    Verify that the user query is converted into an embedding.
    """

    embedding_service = MagicMock()
    vector_store = MagicMock()

    embedding_service.embed_text.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    vector_store.search.return_value = []

    retriever = VectorRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    retriever.retrieve("What is authentication?")

    embedding_service.embed_text.assert_called_once_with("What is authentication?")


def test_query_vector_is_sent_to_qdrant():
    """
    Verify that the generated query vector is passed to
    the vector store.
    """

    embedding_service = MagicMock()
    vector_store = MagicMock()

    query_vector = [
        0.1,
        0.2,
        0.3,
    ]

    embedding_service.embed_text.return_value = query_vector

    vector_store.search.return_value = [create_result()]

    retriever = VectorRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "What is authentication?",
        limit=5,
    )

    vector_store.search.assert_called_once_with(
        query_vector=query_vector,
        limit=5,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-001"
