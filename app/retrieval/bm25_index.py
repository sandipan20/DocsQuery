"""
DocsQuery - BM25 Index Manager

Manages the BM25 retrieval index and its persistence.

Responsibilities:

    DocumentChunks
        ↓
    BM25 index
        ↓
    Persistent storage

On startup:

    Persistent storage
        ↓
    DocumentChunks
        ↓
    BM25 index
"""

from app.ingestion.models import DocumentChunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.bm25_storage import BM25Storage


class BM25Index:
    """
    Manages the searchable BM25 index.
    """

    def __init__(
        self,
        storage: BM25Storage | None = None,
    ):
        """
        Initialize the BM25 index.

        Args:
            storage:
                Optional persistent storage backend.
        """

        self.retriever = BM25Retriever()

        self.storage = storage

    def build(
        self,
        chunks: list[DocumentChunk],
        persist: bool = True,
    ) -> int:
        """
        Build the BM25 index.

        Args:
            chunks:
                Document chunks to index.

            persist:
                Whether the chunks should also be persisted.

        Returns:
            Number of indexed chunks.
        """

        self.retriever.index(chunks)

        if persist and self.storage is not None:
            self.storage.save(chunks)

        return len(chunks)

    def load(self) -> int:
        """
        Load persisted chunks and rebuild the BM25 index.

        Returns:
            Number of loaded chunks.

        Raises:
            FileNotFoundError:
                If no persisted index exists.
        """

        if self.storage is None:
            raise RuntimeError("BM25 storage is not configured.")

        chunks = self.storage.load()

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
