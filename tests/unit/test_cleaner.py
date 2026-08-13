"""
Tests for document text cleaning.
"""

from app.ingestion.cleaner import clean_page, clean_text
from app.ingestion.models import DocumentPage


def test_clean_text_normalizes_spaces():
    """
    Multiple spaces should be reduced to a single space.
    """

    raw_text = "Hello     world"

    cleaned = clean_text(raw_text)

    assert cleaned == "Hello world"


def test_clean_text_normalizes_line_endings():
    """
    Different line-ending formats should be normalized.
    """

    raw_text = "Hello\r\nWorld\rTest"

    cleaned = clean_text(raw_text)

    assert cleaned == "Hello\nWorld\nTest"


def test_clean_text_removes_excessive_blank_lines():
    """
    Multiple blank lines should be reduced to one.
    """

    raw_text = "First paragraph.\n\n\n\nSecond paragraph."

    cleaned = clean_text(raw_text)

    assert cleaned == ("First paragraph.\n\nSecond paragraph.")


def test_clean_page_preserves_metadata():
    """
    Cleaning text must not destroy document metadata.
    """

    page = DocumentPage(
        page_number=42,
        text="Hello     world",
        source="document.pdf",
    )

    cleaned_page = clean_page(page)

    # Metadata must remain unchanged.
    assert cleaned_page.page_number == 42
    assert cleaned_page.source == "document.pdf"

    # Text should be cleaned.
    assert cleaned_page.text == "Hello world"
