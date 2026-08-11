"""
Tests for document ingestion models.
"""

import pytest
from pydantic import ValidationError

from app.ingestion.models import DocumentChunk, DocumentPage


def test_document_page_creation():
    """
    Verify that a valid DocumentPage can be created.
    """

    page = DocumentPage(
        page_number=1,
        text="This is a test page.",
        source="test.pdf",
    )

    assert page.page_number == 1
    assert page.text == "This is a test page."
    assert page.source == "test.pdf"


def test_document_page_rejects_invalid_page_number():
    """
    Page numbers must start at 1.
    """

    with pytest.raises(ValidationError):
        DocumentPage(
            page_number=0,
            text="Invalid page.",
            source="test.pdf",
        )


def test_document_chunk_creation():
    """
    Verify that a valid DocumentChunk can be created.
    """

    chunk = DocumentChunk(
        chunk_id="chunk-001",
        document_id="document-001",
        text="This is a test chunk.",
        source="test.pdf",
        page_number=1,
    )

    assert chunk.chunk_id == "chunk-001"
    assert chunk.document_id == "document-001"
    assert chunk.page_number == 1
