"""
Unit tests for the retrieval index manager.
"""

from unittest.mock import MagicMock

import pytest

from app.ingestion.models import DocumentChunk
from app.retrieval.index_manager import (
    RetrievalIndexManager,
)


def create_chunk() -> DocumentChunk:
    """
    Create a test chunk.
    """

    return DocumentChunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        text="Test content.",
        source="test.pdf",
        page_number=1,
        chunk_index=0,
    )


def test_empty_chunks_return_zero():
    """
    Empty input should not index anything.
    """

    vector_indexer = MagicMock()
    bm25_index = MagicMock()

    manager = RetrievalIndexManager(
        vector_indexer=vector_indexer,
        bm25_index=bm25_index,
    )

    assert manager.index([]) == 0

    vector_indexer.index_chunks.assert_not_called()
    bm25_index.build.assert_not_called()


def test_both_indexes_receive_same_chunks():
    """
    BM25 and vector indexes must receive the same chunks.
    """

    vector_indexer = MagicMock()
    bm25_index = MagicMock()

    vector_indexer.index_chunks.return_value = 1
    bm25_index.build.return_value = 1

    chunks = [create_chunk()]

    manager = RetrievalIndexManager(
        vector_indexer=vector_indexer,
        bm25_index=bm25_index,
    )

    result = manager.index(chunks)

    assert result == 1

    bm25_index.build.assert_called_once_with(chunks)

    vector_indexer.index_chunks.assert_called_once_with(chunks)


def test_mismatched_indexes_raise_error():
    """
    If the two indexes process different numbers of chunks,
    indexing should fail loudly.
    """

    vector_indexer = MagicMock()
    bm25_index = MagicMock()

    vector_indexer.index_chunks.return_value = 2
    bm25_index.build.return_value = 1

    manager = RetrievalIndexManager(
        vector_indexer=vector_indexer,
        bm25_index=bm25_index,
    )

    with pytest.raises(RuntimeError):
        manager.index([create_chunk()])
