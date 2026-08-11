"""
DocsQuery - Document Ingestion Pipeline

This module connects the individual ingestion stages.

Current pipeline:

    PDF
     ↓
    Loader
     ↓
    DocumentPage
     ↓
    Cleaner
     ↓
    Clean DocumentPage
"""

from app.ingestion.cleaner import clean_page
from app.ingestion.loader import load_pdf
from app.ingestion.models import DocumentPage


def ingest_pdf(file_path: str) -> list[DocumentPage]:
    """
    Load and clean a PDF document.

    Args:
        file_path:
            Path to the PDF file.

    Returns:
        A list of cleaned DocumentPage objects.
    """

    # Step 1:
    # Extract pages from the PDF.
    pages = load_pdf(file_path)

    # Step 2:
    # Clean every extracted page.
    cleaned_pages = [
        clean_page(page)
        for page in pages
    ]

    return cleaned_pages