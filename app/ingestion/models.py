"""
DocsQuery - Ingestion Data Models

This module defines the structured data objects used during
document ingestion.

The ingestion pipeline will eventually look like:

    PDF
      ↓
    DocumentPage
      ↓
    DocumentChunk
      ↓
    Embedding
      ↓
    Vector Database
"""

from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """
    Represents one page extracted from a source document.

    Keeping page-level metadata is important because the
    citation system will eventually need to tell the user
    exactly where an answer came from.
    """

    # 1-based page number.
    page_number: int = Field(ge=1)

    # Text extracted from the page.
    text: str

    # Original document filename.
    source: str


class DocumentChunk(BaseModel):
    """
    Represents a searchable chunk created from a document page.

    Chunking will be implemented in the next ingestion stage.
    """

    # Unique identifier for this chunk.
    chunk_id: str

    # ID of the original document.
    document_id: str

    # Text that will eventually be embedded and searched.
    text: str

    # Source document filename.
    source: str

    # Page where this chunk originated.
    page_number: int = Field(ge=1)
