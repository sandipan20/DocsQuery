"""
Tests for the document ingestion pipeline.
"""

from pathlib import Path

import pytest

from app.ingestion.models import DocumentPage
from app.ingestion.pipeline import ingest_pdf


def test_ingest_pdf_returns_chunks():
    """
    Verify that the ingestion pipeline returns cleaned
    DocumentPage objects.
    """

    pdf_path = Path("data/raw/python_documentation.pdf")

    if not pdf_path.exists():
        pytest.fail(f"Sample PDF is missing: {pdf_path}")

    pages = ingest_pdf(str(pdf_path))

    # At least one page should be returned.
    assert len(pages) > 0

    # Every returned item should be a DocumentPage.
    assert all(isinstance(page, DocumentPage) for page in pages)

    # The first page should have valid metadata.
    assert pages[0].page_number == 1
    assert pages[0].source == "python_documentation.pdf"
