"""
DocsQuery - Application Dependency Container

Creates and owns long-lived application dependencies.

Dependencies are initialized once per application process
and reused across requests.
"""

from app.config.settings import get_settings
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_storage import BM25Storage
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_retriever import VectorRetriever
from app.services.retrieval_service import RetrievalService


class AppContainer:
    """
    Holds application-wide services.
    """

    def __init__(self):
        """
        Create application dependencies.
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
        # Cross-encoder reranker
        # ----------------------------------------------------

        self.reranker = CrossEncoderReranker(model_name=settings.reranker_model)

        # ----------------------------------------------------
        # Complete retrieval service
        # ----------------------------------------------------

        self.retrieval_service = RetrievalService(
            bm25_index=self.bm25_index,
            vector_retriever=(self.vector_retriever),
            reranker=self.reranker,
            candidate_limit=20,
            top_k=settings.reranker_top_k,
        )

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
