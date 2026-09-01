"""
DocsQuery - Retrieval Index Manager

Coordinates the indexes required by DocsQuery.

Current indexes:

    1. Qdrant vector index
    2. BM25 keyword index
"""

from app.config.settings import get_settings
from app.ingestion.models import DocumentChunk
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_storage import BM25Storage
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

        settings = get_settings()

        self.vector_indexer = vector_indexer or VectorIndexer()

        # Configure persistent BM25 storage when the caller
        # does not provide a custom BM25 implementation.
        if bm25_index is None:
            storage = BM25Storage(settings.bm25_index_path)

            bm25_index = BM25Index(storage=storage)

        self.bm25_index = bm25_index

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

        # Build and persist BM25.
        bm25_count = self.bm25_index.build(chunks)

        # Build the vector index.
        vector_count = self.vector_indexer.index_chunks(chunks)

        # Both systems must process the same chunks.
        if bm25_count != vector_count:
            raise RuntimeError(
                "BM25 and vector indexes contain different numbers of chunks."
            )

        return vector_count

    def load_bm25(self) -> int:
        """
        Load the persisted BM25 corpus and rebuild the
        in-memory BM25 search index.

        Returns:
            Number of chunks loaded.
        """

        return self.bm25_index.load()

    def rebuild(
        self,
        chunks: list[DocumentChunk],
    ) -> int:
        """
        Completely rebuild BM25 and vector indexes.

        This should be used when the source corpus has changed
        and we want a clean retrieval state.

        Args:
            chunks:
                Complete corpus of document chunks.

        Returns:
            Number of indexed chunks.
        """

        if not chunks:
            raise ValueError("Cannot rebuild indexes from an empty corpus.")

        # ----------------------------------------------------
        # Rebuild BM25
        # ----------------------------------------------------

        if self.bm25_index.storage is None:
            raise RuntimeError("BM25 storage is not configured.")

        self.bm25_index.storage.delete()

        bm25_count = self.bm25_index.build(chunks)

        # ----------------------------------------------------
        # Rebuild Qdrant
        # ----------------------------------------------------

        if not chunks:
            return 0

        # Generate embeddings once so we know the vector size.
        texts = [chunk.text for chunk in chunks]

        embeddings = self.vector_indexer.embedding_service.embed_texts(texts)

        self.vector_indexer.vector_store.recreate_collection(
            vector_size=len(embeddings[0])
        )

        self.vector_indexer.vector_store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        vector_count = len(chunks)

        if bm25_count != vector_count:
            raise RuntimeError(
                "BM25 and vector indexes contain different "
                "numbers of chunks after rebuild."
            )

        return vector_count
