"""
DocsQuery - Corpus Integrity Tests

These tests protect important invariants of the persisted
retrieval corpus.

They are especially important after expanding from a tiny
two-chunk corpus to a larger multi-document corpus.
"""

from app.config.settings import get_settings
from app.retrieval.bm25_storage import BM25Storage


def test_all_persisted_chunk_ids_are_unique():
    """
    Every persisted chunk must have a unique workspace/chunk ID pair.

    Why this matters:

    Qdrant point IDs include the workspace so identical documents
    in different sessions can safely reuse deterministic chunk IDs.
    """

    settings = get_settings()

    storage = BM25Storage(settings.bm25_index_path)

    # The test should not fail simply because the corpus has
    # not been generated yet.
    if not storage.exists():
        return

    chunks = storage.load()

    point_keys = [(chunk.workspace_id, chunk.chunk_id) for chunk in chunks]

    assert len(point_keys) == len(set(point_keys))


def test_chunks_have_required_metadata():
    """
    Every persisted chunk must contain the metadata required
    for retrieval, debugging, and future citations.
    """

    settings = get_settings()

    storage = BM25Storage(settings.bm25_index_path)

    if not storage.exists():
        return

    chunks = storage.load()

    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.document_id
        assert chunk.text.strip()
        assert chunk.source

        # Page numbers are 1-based.
        assert chunk.page_number >= 1

        # Chunk indexes are 0-based.
        assert chunk.chunk_index >= 0
