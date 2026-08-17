"""
Unit tests for the Qdrant vector store.

These tests use a fake client so that unit tests do not
require a running Qdrant server.
"""

from unittest.mock import patch
from uuid import UUID

import pytest

from app.ingestion.models import DocumentChunk
from app.retrieval.vector_store import QdrantVectorStore


def create_chunk() -> DocumentChunk:
    """
    Create a predictable test chunk.
    """

    return DocumentChunk(
        chunk_id="doc-001-chunk-0",
        document_id="doc-001",
        text="This is a test chunk.",
        source="test.pdf",
        page_number=1,
        chunk_index=0,
    )


@patch("app.retrieval.vector_store.QdrantClient")
def test_upsert_rejects_mismatched_lengths(
    mock_client,
):
    """
    Number of chunks and embeddings must match.
    """

    store = QdrantVectorStore()

    with pytest.raises(ValueError):
        store.upsert_chunks(
            chunks=[create_chunk()],
            embeddings=[],
        )


@patch("app.retrieval.vector_store.QdrantClient")
def test_empty_upsert_does_nothing(
    mock_client,
):
    """
    An empty chunk list should not make a Qdrant request.
    """

    store = QdrantVectorStore()

    store.upsert_chunks(
        chunks=[],
        embeddings=[],
    )

    store.client.upsert.assert_not_called()


@patch("app.retrieval.vector_store.QdrantClient")
def test_upsert_creates_qdrant_point(
    mock_client,
):
    """
    Verify that chunks are converted into Qdrant points.
    """

    fake_client = mock_client.return_value

    # Pretend that no collections currently exist.
    fake_client.get_collections.return_value.collections = []

    store = QdrantVectorStore()

    chunk = create_chunk()

    embedding = [
        0.1,
        0.2,
        0.3,
    ]

    store.upsert_chunks(
        chunks=[chunk],
        embeddings=[embedding],
    )

    # Verify that Qdrant received one point.
    fake_client.upsert.assert_called_once()

    call = fake_client.upsert.call_args

    points = call.kwargs["points"]

    assert len(points) == 1

    # Qdrant receives a valid UUID as its database point ID.
    UUID(points[0].id)

    # Our original application-level chunk ID remains
    # available inside the payload.
    assert points[0].payload["chunk_id"] == "doc-001-chunk-0"

    assert points[0].payload["document_id"] == "doc-001"

    assert points[0].payload["page_number"] == 1
