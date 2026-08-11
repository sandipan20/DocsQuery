"""
DocsQuery - PDF Document Loader

This module reads PDF files and converts each page into a
structured DocumentPage object.
"""

from pathlib import Path

from pypdf import PdfReader

from app.ingestion.models import DocumentPage


def load_pdf(file_path: str) -> list[DocumentPage]:
    """
    Load a PDF and extract its pages.

    Args:
        file_path:
            Path to the PDF file.

    Returns:
        A list of DocumentPage objects.

    Raises:
        FileNotFoundError:
            If the PDF does not exist.

        ValueError:
            If the supplied file is not a PDF.
    """

    # Convert the supplied string into a Path object.
    path = Path(file_path)

    # Verify that the file exists.
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    # Verify that the file has a PDF extension.
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    # Open the PDF.
    reader = PdfReader(str(path))

    pages: list[DocumentPage] = []

    # Extract every page separately.
    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        # Extract text. Some PDFs may return None.
        text = page.extract_text() or ""

        # Convert the extracted page into our structured model.
        document_page = DocumentPage(
            page_number=page_number,
            text=text.strip(),
            source=path.name,
        )

        pages.append(document_page)

    return pages
