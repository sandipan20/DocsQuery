"""
DocsQuery - Document Chunker

Converts cleaned DocumentPage objects into smaller
DocumentChunk objects suitable for retrieval.

Initial strategy:
    - 500 words per chunk
    - 50 words overlap

The chunker preserves document and page metadata so that
retrieved chunks can later be traced back to their source.
"""

from app.ingestion.models import DocumentChunk, DocumentPage


def chunk_page(
    page: DocumentPage,
    document_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """
    Split one document page into overlapping chunks.

    Args:
        page:
            Cleaned document page.

        document_id:
            Stable document identifier.

        chunk_size:
            Maximum number of words per chunk.

        overlap:
            Number of words shared between adjacent chunks.

    Returns:
        List of DocumentChunk objects.

    Note:
        chunk_index starts at zero for this page only.
        Global chunk IDs are assigned by chunk_pages().
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = page.text.split()

    if not words:
        return []

    chunks: list[DocumentChunk] = []

    step = chunk_size - overlap

    chunk_index = 0
    start = 0

    while start < len(words):
        chunk_words = words[start : start + chunk_size]

        chunk_text = " ".join(chunk_words)

        chunks.append(
            DocumentChunk(
                # Temporary ID.
                #
                # chunk_pages() will assign the final globally
                # unique ID after combining all pages.
                chunk_id="",
                document_id=document_id,
                text=chunk_text,
                source=page.source,
                page_number=page.page_number,
                chunk_index=chunk_index,
            )
        )

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
    Chunk all pages and assign globally unique IDs.

    Global indexing is important because chunk IDs must be
    unique across the entire document.
    """

    chunks: list[DocumentChunk] = []

    for page in pages:
        page_chunks = chunk_page(
            page=page,
            document_id=document_id,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        chunks.extend(page_chunks)

    # --------------------------------------------------------
    # Assign global chunk positions and IDs.
    #
    # Example:
    #
    # document-hash-chunk-0
    # document-hash-chunk-1
    # document-hash-chunk-2
    #
    # regardless of which page they came from.
    # --------------------------------------------------------

    for global_index, chunk in enumerate(chunks):
        chunk.chunk_index = global_index

        chunk.chunk_id = f"{document_id}-chunk-{global_index}"

    return chunks
