"""
Unit tests for BM25 persistence.
"""

from pathlib import Path

import pytest

from app.ingestion.models import DocumentChunk
from app.retrieval.bm25_storage import BM25Storage


def create_chunks() -> list[DocumentChunk]:
    """
    Create predictable test chunks.
    """

    return [
        DocumentChunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            text="Python is a programming language.",
            source="test.pdf",
            page_number=1,
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="chunk-002",
            document_id="doc-001",
            text="Qdrant is a vector database.",
            source="test.pdf",
            page_number=2,
            chunk_index=1,
        ),
    ]


def test_storage_file_does_not_exist_initially(
    tmp_path: Path,
):
    """
    A new storage location should not report an existing file.
    """

    storage = BM25Storage(str(tmp_path / "bm25.json"))

    assert storage.exists() is False


def test_save_creates_storage_file(
    tmp_path: Path,
):
    """
    Saving chunks should create the JSON file.
    """

    storage = BM25Storage(str(tmp_path / "bm25.json"))

    storage.save(create_chunks())

    assert storage.exists() is True


def test_save_and_load_preserves_chunks(
    tmp_path: Path,
):
    """
    Saved chunks should be identical after loading.
    """

    storage = BM25Storage(str(tmp_path / "bm25.json"))

    original = create_chunks()

    storage.save(original)

    loaded = storage.load()

    assert loaded == original


def test_load_missing_file_raises_error(
    tmp_path: Path,
):
    """
    Loading a missing persistence file should fail clearly.
    """

    storage = BM25Storage(str(tmp_path / "missing.json"))

    with pytest.raises(FileNotFoundError):
        storage.load()
