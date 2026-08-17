"""
DocsQuery - BM25 Keyword Retriever

Provides traditional keyword-based retrieval using BM25.

BM25 is useful for:
- exact terminology
- identifiers
- names
- error messages
- technical keywords

Current architecture:

    DocumentChunk
        ↓
    BM25 Index
        ↓
    Query
        ↓
    Ranked RetrievalResult
"""

import re

from rank_bm25 import BM25Okapi

from app.ingestion.models import DocumentChunk
from app.retrieval.models import RetrievalResult


def tokenize(text: str) -> list[str]:
    """
    Convert text into simple lowercase tokens.

    Args:
        text:
            Input text.

    Returns:
        List of normalized tokens.
    """

    # Lowercase the text so searches are case-insensitive.
    text = text.lower()

    # Extract words and numbers.
    #
    # Example:
    #
    # "Python 3.12 is great!"
    #
    # becomes approximately:
    #
    # ["python", "3", "12", "is", "great"]
    return re.findall(
        r"\b\w+\b",
        text,
    )


class BM25Retriever:
    """
    In-memory BM25 keyword retriever.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk] | None = None,
    ):
        """
        Initialize the BM25 retriever.

        Args:
            chunks:
                Optional initial document chunks.
        """

        self.chunks: list[DocumentChunk] = []

        self.bm25: BM25Okapi | None = None

        if chunks:
            self.index(chunks)

    def index(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """
        Build the BM25 index from document chunks.

        Args:
            chunks:
                Chunks that should become searchable.
        """

        if not chunks:
            self.chunks = []
            self.bm25 = None
            return

        self.chunks = list(chunks)

        # Tokenize every chunk.
        tokenized_documents = [tokenize(chunk.text) for chunk in self.chunks]

        # Build the BM25 index.
        self.bm25 = BM25Okapi(tokenized_documents)

    def retrieve(
        self,
        query: str,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """
        Retrieve chunks using BM25.

        Args:
            query:
                User's search query.

            limit:
                Maximum number of results.

        Returns:
            BM25-ranked RetrievalResult objects.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        # We cannot search until an index exists.
        if self.bm25 is None:
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        # Calculate BM25 scores for every indexed chunk.
        scores = self.bm25.get_scores(query_tokens)

        # Sort indexes by descending score.
        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[RetrievalResult] = []

        for index in ranked_indexes[:limit]:
            chunk = self.chunks[index]

            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    source=chunk.source,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    score=float(scores[index]),
                )
            )

        return results
