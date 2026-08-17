"""
DocsQuery - Retrieval Index Manager

Coordinates the indexes required by DocsQuery.

Current indexes:

    1. Qdrant vector index
    2. BM25 keyword index

The manager keeps the two retrieval systems synchronized
during indexing.
"""

from app.ingestion.models import DocumentChunk
from app.retrieval.bm25_index import BM25Index
from app.retrieval.indexer import VectorIndexer


class RetrievalIndexManager:
    """
    Coordinates vector and BM25 indexing.
    """

    def __init__(
        self,
        vector_indexer: VectorIndexer | None = None,
        bm25_index: BM25Index | None = None,
    ):
        """
        Initialize the index manager.
        """

        self.vector_indexer = vector_indexer or VectorIndexer()

        self.bm25_index = bm25_index or BM25Index()

    def index(
        self,
        chunks: list[DocumentChunk],
    ) -> int:
        """
        Index chunks into both retrieval systems.

        Args:
            chunks:
                Document chunks.

        Returns:
            Number of chunks indexed.
        """

        if not chunks:
            return 0

        # Build the keyword index.
        bm25_count = self.bm25_index.build(chunks)

        # Build the vector index.
        vector_count = self.vector_indexer.index_chunks(chunks)

        # Both indexes should process the same number
        # of chunks.
        if bm25_count != vector_count:
            raise RuntimeError(
                "BM25 and vector indexes contain different numbers of chunks."
            )

        return vector_count
