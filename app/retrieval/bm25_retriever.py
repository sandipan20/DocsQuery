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
    Workspace Filter
        ↓
    Ranked RetrievalResult
"""

import math
import re

from rank_bm25 import BM25Okapi

from app.ingestion.models import DocumentChunk
from app.retrieval.models import RetrievalResult


def compute_bm25_scores(
    tokenized_documents: list[list[str]],
    query_tokens: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """
    Compute non-negative BM25 scores for tokenized documents.

    Uses Robertson-Spärck-Jones / Lucene non-negative IDF so that
    query terms present in small user-uploaded corpora (e.g. 1-5 chunks)
    always produce strictly positive scores rather than negative IDFs.
    """
    corpus_size = len(tokenized_documents)
    if corpus_size == 0 or not query_tokens:
        return [0.0] * corpus_size

    total_len = sum(len(doc) for doc in tokenized_documents)
    avgdl = (total_len / corpus_size) if corpus_size > 0 else 1.0

    doc_freqs: list[dict[str, int]] = []
    for doc in tokenized_documents:
        freqs: dict[str, int] = {}
        for token in doc:
            freqs[token] = freqs.get(token, 0) + 1
        doc_freqs.append(freqs)

    scores = [0.0] * corpus_size
    doc_lens = [len(doc) for doc in tokenized_documents]

    for q_token in query_tokens:
        n = sum(1 for df in doc_freqs if q_token in df)
        if n == 0:
            continue

        idf = max(0.0, math.log(1.0 + (corpus_size - n + 0.5) / (n + 0.5)))
        for i, df in enumerate(doc_freqs):
            tf = df.get(q_token, 0)
            if tf > 0:
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (doc_lens[i] / (avgdl or 1.0)))
                scores[i] += idf * (numerator / denominator)

    return scores

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
        workspace_id: str = "public",
        document_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve chunks using BM25.

        Args:
            query:
                User's search query.

            limit:
                Maximum number of results.

            workspace_id:
                Workspace whose documents are allowed to appear
                in the results.

        Returns:
            BM25-ranked RetrievalResult objects belonging only
            to the requested workspace.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        # Searching is impossible until chunks have been indexed.
        if not self.chunks:
            return []

        # Apply the exact same preprocessing to the query
        # that was applied to the documents.
        query_tokens = tokenize(query)

        # A query containing only stopwords has no useful
        # lexical signal for BM25.
        if not query_tokens:
            return []

        # Select the authorized corpus before computing BM25 statistics.
        scoped_chunks = [
            chunk
            for chunk in self.chunks
            if chunk.workspace_id == workspace_id
            and (document_ids is None or chunk.document_id in document_ids)
        ]
        if not scoped_chunks:
            return []

        tokenized_scoped = [tokenize(chunk.text) for chunk in scoped_chunks]
        scores = compute_bm25_scores(tokenized_scoped, query_tokens)

        ranked_indexes = sorted(
            range(len(scoped_chunks)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[RetrievalResult] = []

        # Convert the highest-ranked workspace-owned chunks
        # into the common RetrievalResult model used by the
        # rest of DocsQuery.
        for index in ranked_indexes[:limit]:
            chunk = scoped_chunks[index]

            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    workspace_id=chunk.workspace_id,
                    text=chunk.text,
                    source=chunk.source,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    score=float(scores[index]),
                )
            )

        return results
