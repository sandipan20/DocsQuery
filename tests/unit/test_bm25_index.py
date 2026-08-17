"""
Tests for the BM25 index manager.
"""

from pathlib import Path

from app.ingestion.models import DocumentChunk
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_storage import BM25Storage


def create_chunks() -> list[DocumentChunk]:
    """
    Create predictable test chunks.
    """

    return [
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


def test_build_returns_chunk_count():
    """
    Building the index should return the number of chunks.
    """

    index = BM25Index()

    count = index.build(
        create_chunks(),
        persist=False,
    )

    assert count == 2


def test_search_uses_built_index():
    """
    Verify that the index can retrieve an indexed chunk.
    """

    index = BM25Index()

    index.build(
        create_chunks(),
        persist=False,
    )

    results = index.search(
        "Python",
        limit=1,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"


def test_index_can_be_saved_and_loaded(
    tmp_path: Path,
):
    """
    Verify that a BM25 index can be reconstructed from
    persistent storage.
    """

    storage = BM25Storage(str(tmp_path / "bm25.json"))

    original_index = BM25Index(storage=storage)

    original_index.build(create_chunks())

    # Create a completely new index object.
    loaded_index = BM25Index(storage=storage)

    count = loaded_index.load()

    assert count == 2

    results = loaded_index.search(
        "Python",
        limit=1,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
