"""
DocsQuery - Document Ingestion Pipeline

Current pipeline:

    PDF
     ↓
    Metadata
     ↓
    Loader
     ↓
    DocumentPage
     ↓
    Cleaner
     ↓
    Clean DocumentPage
     ↓
    Chunker
     ↓
    DocumentChunk
"""

from app.ingestion.chunker import chunk_pages
from app.ingestion.cleaner import clean_page
from app.ingestion.loader import load_pdf
from app.ingestion.metadata import generate_document_id
from app.ingestion.models import DocumentChunk


def ingest_pdf(
    file_path: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """
    Run the complete document ingestion pipeline.

    The document ID is generated automatically from the
    document contents.

    Args:
        file_path:
            Path to the PDF.

        chunk_size:
            Maximum words per chunk.

        overlap:
            Number of overlapping words.

    Returns:
        Searchable DocumentChunk objects.
    """

    # --------------------------------------------------------
    # Stage 1:
    # Generate a deterministic document ID.
    # --------------------------------------------------------

    document_id = generate_document_id(file_path)

    # --------------------------------------------------------
    # Stage 2:
    # Extract pages from the PDF.
    # --------------------------------------------------------

    pages = load_pdf(file_path)

    # --------------------------------------------------------
    # Stage 3:
    # Clean extracted text.
    # --------------------------------------------------------

    cleaned_pages = [clean_page(page) for page in pages]

    # --------------------------------------------------------
    # Stage 4:
    # Create searchable chunks.
    # --------------------------------------------------------

    chunks = chunk_pages(
        pages=cleaned_pages,
        document_id=document_id,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return chunks
