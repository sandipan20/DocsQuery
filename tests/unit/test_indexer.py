"""
Unit tests for the vector indexing service.
"""

from unittest.mock import MagicMock

from app.ingestion.models import DocumentChunk
from app.retrieval.indexer import VectorIndexer


def create_chunk(
    chunk_id: str,
    text: str,
) -> DocumentChunk:
    """
    Create a predictable test chunk.
    """

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        text=text,
        source="test.pdf",
        page_number=1,
        chunk_index=0,
    )


def test_empty_chunks_return_zero():
    """
    Indexing an empty list should do nothing.
    """

    embedding_service = MagicMock()
    vector_store = MagicMock()

    indexer = VectorIndexer(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    result = indexer.index_chunks([])

    assert result == 0

    embedding_service.embed_texts.assert_not_called()
    vector_store.upsert_chunks.assert_not_called()


def test_index_chunks_generates_embeddings():
    """
    Verify that chunk text is sent to the embedding service.
    """

    embedding_service = MagicMock()
    vector_store = MagicMock()

    embedding_service.embed_texts.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    chunks = [
        create_chunk(
            "chunk-1",
            "First document chunk.",
        ),
        create_chunk(
            "chunk-2",
            "Second document chunk.",
        ),
    ]

    indexer = VectorIndexer(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    result = indexer.index_chunks(chunks)

    assert result == 2

    embedding_service.embed_texts.assert_called_once_with(
        [
            "First document chunk.",
            "Second document chunk.",
        ]
    )


def test_index_chunks_stores_embeddings():
    """
    Verify that generated embeddings are sent to Qdrant.
    """

    embedding_service = MagicMock()
    vector_store = MagicMock()

    embeddings = [
        [0.1, 0.2, 0.3],
    ]

    embedding_service.embed_texts.return_value = embeddings

    chunks = [
        create_chunk(
            "chunk-1",
            "Test document chunk.",
        )
    ]

    indexer = VectorIndexer(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    indexer.index_chunks(chunks)

    vector_store.upsert_chunks.assert_called_once_with(
        chunks=chunks,
        embeddings=embeddings,
    )