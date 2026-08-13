"""
DocsQuery - Document Chunker

Converts cleaned DocumentPage objects into smaller
DocumentChunk objects suitable for retrieval.

Initial strategy:
    - 500 words per chunk
    - 50 words overlap

These values are starting points, not permanent choices.
They will eventually be evaluated using our RAG evaluation
pipeline.
"""

from app.ingestion.models import DocumentChunk, DocumentPage


def chunk_page(
    page: DocumentPage,
    document_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """
    Split a document page into overlapping word-based chunks.

    Args:
        page:
            Cleaned DocumentPage.

        document_id:
            Unique identifier for the source document.

        chunk_size:
            Maximum number of words in each chunk.

        overlap:
            Number of words repeated between adjacent chunks.

    Returns:
        A list of DocumentChunk objects.

    Raises:
        ValueError:
            If chunk_size or overlap has an invalid value.
    """

    # --------------------------------------------------------
    # Validate chunk configuration.
    # --------------------------------------------------------

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # --------------------------------------------------------
    # Empty pages do not produce chunks.
    # --------------------------------------------------------

    words = page.text.split()

    if not words:
        return []

    chunks: list[DocumentChunk] = []

    # The next chunk starts after this many new words.
    step = chunk_size - overlap

    chunk_index = 0
    start = 0

    while start < len(words):
        # Select the words belonging to this chunk.
        chunk_words = words[start : start + chunk_size]

        # Convert words back into readable text.
        chunk_text = " ".join(chunk_words)

        chunks.append(
            DocumentChunk(
                chunk_id=f"{document_id}-chunk-{chunk_index}",
                document_id=document_id,
                text=chunk_text,
                source=page.source,
                page_number=page.page_number,
                chunk_index=chunk_index,
            )
        )

        # Move forward while preserving the overlap.
        start += step
        chunk_index += 1

    return chunks


def chunk_pages(
    pages: list[DocumentPage],
    document_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """
    Chunk all pages belonging to a document.

    Args:
        pages:
            Cleaned document pages.

        document_id:
            Unique identifier for the document.

        chunk_size:
            Maximum words per chunk.

        overlap:
            Number of overlapping words.

    Returns:
        A flat list of DocumentChunk objects.
    """

    chunks: list[DocumentChunk] = []

    # Process every page independently.
    for page in pages:
        page_chunks = chunk_page(
            page=page,
            document_id=document_id,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        chunks.extend(page_chunks)

    # Re-index chunks globally so chunk_index represents their
    # position in the complete document.
    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index

    return chunks
