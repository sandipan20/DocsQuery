"""
Tests for the DocsQuery PDF loader.
"""

from pathlib import Path

import pytest

from app.ingestion.loader import load_pdf
from app.ingestion.models import DocumentPage


def test_pdf_file_must_exist():
    """
    The loader should raise FileNotFoundError when the
    supplied PDF does not exist.
    """

    with pytest.raises(FileNotFoundError):
        load_pdf("does-not-exist.pdf")


def test_file_must_be_pdf(tmp_path: Path):
    """
    The loader should reject files that are not PDFs.
    """

    # Create a temporary text file.
    text_file = tmp_path / "example.txt"
    text_file.write_text("This is not a PDF.")

    with pytest.raises(ValueError):
        load_pdf(str(text_file))


def test_load_real_pdf():
    """
    Verify that the loader successfully extracts a real PDF.
    """

    pdf_path = Path("data/raw/python_documentation.pdf")

    if not pdf_path.exists():
        pytest.fail(f"Sample PDF is missing: {pdf_path}")

    pages = load_pdf(str(pdf_path))

    # The PDF should contain at least one page.
    assert len(pages) > 0

    first_page = pages[0]

    # The loader should return our Pydantic model.
    assert isinstance(first_page, DocumentPage)

    # Verify page metadata.
    assert first_page.page_number == 1

    # Verify source metadata.
    assert first_page.source == "python_documentation.pdf"

    # Verify extracted text.
    assert isinstance(first_page.text, str)
