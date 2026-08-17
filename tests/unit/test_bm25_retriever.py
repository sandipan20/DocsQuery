"""
Unit tests for the BM25 retriever.
"""

import pytest

from app.ingestion.models import DocumentChunk
from app.retrieval.bm25_retriever import (
    BM25Retriever,
    tokenize,
)


def create_chunks() -> list[DocumentChunk]:
    """
    Create predictable chunks for BM25 tests.
    """

    return [
        DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text=("Python is a programming language."),
            source="python.pdf",
            page_number=1,
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            text=("Qdrant is a vector database."),
            source="python.pdf",
            page_number=2,
            chunk_index=1,
        ),
        DocumentChunk(
            chunk_id="chunk-3",
            document_id="doc-1",
            text=("BM25 is a keyword retrieval algorithm."),
            source="python.pdf",
            page_number=3,
            chunk_index=2,
        ),
    ]


def test_tokenize_normalizes_text():
    """
    Tokenization should lowercase the input.
    """

    tokens = tokenize("Python IS Great!")

    assert tokens == [
        "python",
        "is",
        "great",
    ]


def test_empty_index_returns_no_results():
    """
    Searching an empty index should return no results.
    """

    retriever = BM25Retriever()

    results = retriever.retrieve("Python")

    assert results == []


def test_empty_query_is_rejected():
    """
    Empty queries should be rejected.
    """

    retriever = BM25Retriever(create_chunks())

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_invalid_limit_is_rejected():
    """
    Retrieval limit must be positive.
    """

    retriever = BM25Retriever(create_chunks())

    with pytest.raises(ValueError):
        retriever.retrieve(
            "Python",
            limit=0,
        )


def test_bm25_returns_matching_document():
    """
    BM25 should return the document containing the searched
    keyword.
    """

    retriever = BM25Retriever(create_chunks())

    results = retriever.retrieve(
        "Python",
        limit=1,
    )

    assert len(results) == 1

    assert results[0].chunk_id == "chunk-1"
    assert results[0].source == "python.pdf"
    assert results[0].page_number == 1


def test_bm25_respects_limit():
    """
    BM25 should return no more than the requested number
    of results.
    """

    retriever = BM25Retriever(create_chunks())

    results = retriever.retrieve(
        "database programming algorithm",
        limit=2,
    )

    assert len(results) == 2
