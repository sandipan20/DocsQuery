"""
DocsQuery - Retrieval Models

Defines the structured objects returned by retrieval systems.

Keeping retrieval results independent from Qdrant means we can
later replace or add retrieval backends without changing the
rest of the application.
"""

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """
    Represents one document chunk returned by a retriever.
    """

    # The chunk that was retrieved.
    chunk_id: str

    # Source document identifier.
    document_id: str

    # Actual text used by the LLM.
    text: str

    # Original document filename.
    source: str

    # Page supporting the retrieved content.
    page_number: int = Field(ge=1)

    # Position of the chunk in the document.
    chunk_index: int = Field(ge=0)

    # Similarity/relevance score.
    score: float
