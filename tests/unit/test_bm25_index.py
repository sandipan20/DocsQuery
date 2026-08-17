"""
Tests for the BM25 index manager.
"""

from app.ingestion.models import DocumentChunk
from app.retrieval.bm25_index import BM25Index


def test_build_returns_chunk_count():
    """
    Building the index should return the number of chunks.
    """

    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="Python programming.",
            source="test.pdf",
            page_number=1,
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            text="Vector databases.",
            source="test.pdf",
            page_number=2,
            chunk_index=1,
        ),
    ]

    index = BM25Index()

    count = index.build(chunks)

    assert count == 2


def test_search_uses_built_index():
    """
    Verify that the index can retrieve an indexed chunk.
    """

    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="Python programming.",
            source="test.pdf",
            page_number=1,
            chunk_index=0,
        ),
    ]

    index = BM25Index()

    index.build(chunks)

    results = index.search(
        "Python",
        limit=1,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
