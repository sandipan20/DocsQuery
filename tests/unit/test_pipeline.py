"""
Tests for the document ingestion pipeline.
"""

from pathlib import Path

import pytest

from app.ingestion.models import DocumentChunk
from app.ingestion.pipeline import ingest_pdf


def test_ingest_pdf_returns_chunks():
    """
    Verify that the complete ingestion pipeline produces
    DocumentChunk objects with an automatically generated
    document ID.
    """

    pdf_path = Path("data/raw/python_documentation.pdf")

    if not pdf_path.exists():
        pytest.fail(f"Sample PDF is missing: {pdf_path}")

    chunks = ingest_pdf(
        file_path=str(pdf_path),
    )

    # A real document should produce at least one chunk.
    assert len(chunks) > 0

    # Every output should be a DocumentChunk.
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)

    # The document ID should be generated automatically.
    assert len(chunks[0].document_id) == 64

    # Verify citation metadata.
    assert chunks[0].page_number >= 1
    assert chunks[0].source == "python_documentation.pdf"

    # Verify the chunk ID contains the document ID.
    assert chunks[0].chunk_id.startswith(f"{chunks[0].document_id}-chunk-")


def test_ingest_pdf_accepts_custom_chunk_configuration():
    """
    Verify that custom chunk size and overlap values are
    passed through the ingestion pipeline.
    """

    pdf_path = Path("data/raw/python_documentation.pdf")

    if not pdf_path.exists():
        pytest.fail(f"Sample PDF is missing: {pdf_path}")

    chunks = ingest_pdf(
        file_path=str(pdf_path),
        chunk_size=100,
        overlap=20,
    )

    # The pipeline should still produce chunks.
    assert len(chunks) > 0

    # Every chunk should contain no more than 100 words.
    assert all(len(chunk.text.split()) <= 100 for chunk in chunks)
