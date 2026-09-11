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

# ------------------------------------------------------------
# Common English stopwords.
#
# These words usually provide little useful signal for
# technical-document retrieval.
#
# Important technical terms such as:
#
#     git
#     branch
#     merge
#     rebase
#     python
#
# are intentionally NOT included.
# ------------------------------------------------------------

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def tokenize(text: str) -> list[str]:
    """
    Convert text into normalized BM25 tokens.

    Processing steps:

    1. Convert text to lowercase.
    2. Extract word/number tokens.
    3. Remove common English stopwords.

    Example:

        "How do I create a new Git branch?"

    becomes approximately:

        ["create", "new", "git", "branch"]

    Technical terms are preserved because only the explicit
    stopword set above is removed.
    """

    # Make matching case-insensitive.
    text = text.lower()

    # Extract words and numbers.
    tokens = re.findall(
        r"\b\w+\b",
        text,
    )

    # Remove only the explicitly defined English stopwords.
    return [token for token in tokens if token not in STOPWORDS]


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

        # Keep the original DocumentChunk objects because their
        # metadata is required when constructing RetrievalResult.
        self.chunks = list(chunks)

        # Tokenize every document using the same preprocessing
        # function used for queries.
        tokenized_documents = [tokenize(chunk.text) for chunk in self.chunks]

        # Build the BM25Okapi index.
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

        # Searching is impossible until an index has been built.
        if self.bm25 is None:
            return []

        # Apply the exact same preprocessing to the query
        # that was applied to the documents.
        query_tokens = tokenize(query)

        # A query containing only stopwords has no useful
        # lexical signal for BM25.
        if not query_tokens:
            return []

        # Calculate one BM25 score for every indexed chunk.
        scores = self.bm25.get_scores(query_tokens)

        # Sort document indexes by descending BM25 score.
        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[RetrievalResult] = []

        # Convert the highest-ranked chunks into the common
        # RetrievalResult model used by the rest of DocsQuery.
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
