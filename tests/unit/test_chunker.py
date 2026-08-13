"""
Tests for document chunking.
"""

import pytest

from app.ingestion.chunker import chunk_page, chunk_pages
from app.ingestion.models import DocumentPage


def create_page(word_count: int = 10) -> DocumentPage:
    """
    Create a predictable test page containing numbered words.
    """

    words = [f"word{i}" for i in range(word_count)]

    return DocumentPage(
        page_number=1,
        text=" ".join(words),
        source="test.pdf",
    )


def test_empty_page_returns_no_chunks():
    """
    An empty page should not produce empty chunks.
    """

    page = DocumentPage(
        page_number=1,
        text="",
        source="test.pdf",
    )

    chunks = chunk_page(
        page=page,
        document_id="doc-1",
    )

    assert chunks == []


def test_small_page_creates_one_chunk():
    """
    A page smaller than the chunk size should produce one chunk.
    """

    page = create_page(word_count=10)

    chunks = chunk_page(
        page=page,
        document_id="doc-1",
        chunk_size=20,
        overlap=5,
    )

    assert len(chunks) == 1

    assert chunks[0].text == (
        "word0 word1 word2 word3 word4 word5 word6 word7 word8 word9"
    )

    assert chunks[0].page_number == 1
    assert chunks[0].source == "test.pdf"
    assert chunks[0].document_id == "doc-1"


def test_chunking_creates_overlap():
    """
    Adjacent chunks should share the configured number of words.
    """

    page = create_page(word_count=10)

    chunks = chunk_page(
        page=page,
        document_id="doc-1",
        chunk_size=6,
        overlap=2,
    )

    # Ten words with a chunk size of six and a step of four
    # produces three chunks:
    #
    # Chunk 0: word0 ... word5
    # Chunk 1: word4 ... word9
    # Chunk 2: word8 ... word9
    assert len(chunks) == 3

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()

    # The final two words of the first chunk should appear
    # at the beginning of the second chunk.
    assert first_words[-2:] == second_words[:2]


def test_invalid_chunk_configuration():
    """
    Invalid chunk parameters should raise ValueError.
    """

    page = create_page()

    with pytest.raises(ValueError):
        chunk_page(
            page=page,
            document_id="doc-1",
            chunk_size=0,
        )

    with pytest.raises(ValueError):
        chunk_page(
            page=page,
            document_id="doc-1",
            chunk_size=10,
            overlap=10,
        )


def test_chunk_pages_preserves_metadata():
    """
    Chunking multiple pages should preserve page and source metadata.
    """

    pages = [
        DocumentPage(
            page_number=1,
            text="page one content",
            source="test.pdf",
        ),
        DocumentPage(
            page_number=2,
            text="page two content",
            source="test.pdf",
        ),
    ]

    chunks = chunk_pages(
        pages=pages,
        document_id="doc-123",
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) == 2

    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2

    assert all(chunk.document_id == "doc-123" for chunk in chunks)
