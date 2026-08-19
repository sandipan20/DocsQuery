"""
DocsQuery - Application Dependency Container

Creates and owns long-lived application dependencies.

These dependencies are initialized once when the application
starts and reused across requests.
"""

from app.config.settings import get_settings
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_storage import BM25Storage
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.services.retrieval_service import RetrievalService


class AppContainer:
    """
    Holds application-wide services.
    """

    def __init__(self):
        """
        Create the dependency container.
        """

        settings = get_settings()

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        bm25_storage = BM25Storage(settings.bm25_index_path)

        self.bm25_index = BM25Index(storage=bm25_storage)

        # ----------------------------------------------------
        # Vector retrieval
        # ----------------------------------------------------

        self.vector_retriever = VectorRetriever()

        # ----------------------------------------------------
        # Hybrid retrieval
        # ----------------------------------------------------

        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=self.bm25_index.retriever,
            vector_retriever=self.vector_retriever,
        )

        # ----------------------------------------------------
        # Application service
        # ----------------------------------------------------

        self.retrieval_service = RetrievalService(
            retriever=self.hybrid_retriever,
        )

        # Tracks whether the BM25 index was successfully
        # loaded during application startup.
        self.bm25_loaded = False

    def load_indexes(self) -> int:
        """
        Load persistent retrieval indexes.

        Returns:
            Number of BM25 chunks loaded.
        """

        if not self.bm25_index.storage.exists():
            return 0

        count = self.bm25_index.load()

        self.bm25_loaded = count > 0

        return count
