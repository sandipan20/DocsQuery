"""
DocsQuery - BM25 Index Manager

Responsible for building and managing the BM25 index.

This is kept separate from BM25Retriever so that later we
can replace the in-memory implementation with persistent
storage.
"""

from app.ingestion.models import DocumentChunk
from app.retrieval.bm25_retriever import BM25Retriever


class BM25Index:
    """
    Manages the searchable BM25 index.
    """

    def __init__(self):
        """
        Create an empty BM25 index.
        """

        self.retriever = BM25Retriever()

    def build(
        self,
        chunks: list[DocumentChunk],
    ) -> int:
        """
        Build the BM25 index.

        Args:
            chunks:
                Document chunks to index.

        Returns:
            Number of indexed chunks.
        """

        self.retriever.index(chunks)

        return len(chunks)

    def search(
        self,
        query: str,
        limit: int = 10,
    ):
        """
        Search the BM25 index.
        """

        return self.retriever.retrieve(
            query=query,
            limit=limit,
        )
