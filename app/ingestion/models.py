"""
DocsQuery - Ingestion Data Models

Defines the structured objects used throughout document
ingestion.
"""

from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """
    Represents one page extracted from a source document.
    """

    # 1-based page number.
    page_number: int = Field(ge=1)

    # Cleaned/extracted page text.
    text: str

    # Original document filename.
    source: str


class DocumentChunk(BaseModel):
    """
    Represents a searchable piece of a document.

    A chunk is created from one or more portions of a page.
    Metadata is preserved so that retrieval results can later
    be converted into citations.
    """

    # Unique identifier for this chunk.
    chunk_id: str

    # Identifier shared by all chunks from the same document.
    document_id: str

    # Identifies which workspace owns this chunk.
    #
    # "public" is used for the existing DocsQuery corpus.
    # Later, uploaded PDFs will use the anonymous session ID.
    workspace_id: str = "public"

    # Searchable chunk text.
    text: str

    # Original document filename.
    source: str

    # Page from which this chunk originated.
    page_number: int = Field(ge=1)

    # Position of the chunk within the document.
    chunk_index: int = Field(ge=0)

    # Total pages in the source document, retained for listing metadata.
    document_page_count: int | None = Field(default=None, ge=1)
