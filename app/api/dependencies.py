"""
DocsQuery - API Dependencies

Provides application-level dependencies to FastAPI routes.

The retrieval service contains:

    BM25
    Vector Retrieval
    Qdrant
    Hybrid RRF

We create it once and reuse it rather than rebuilding the
retrieval stack for every HTTP request.
"""

from functools import lru_cache

from app.config.settings import get_settings
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_storage import BM25Storage
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.services.retrieval_service import RetrievalService


@lru_cache
def get_retrieval_service() -> RetrievalService:
    """
    Create and cache the application's retrieval service.

    The first call creates the retrieval stack.

    Later calls return the same instance.

    This prevents expensive objects such as embedding models
    from being recreated for every HTTP request.
    """

    settings = get_settings()

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    # Create persistent BM25 storage using the configured
    # path from application settings.
    bm25_storage = BM25Storage(settings.bm25_index_path)

    # Create the BM25 index manager.
    bm25_index = BM25Index(storage=bm25_storage)

    # If a persisted BM25 index exists, load it into memory.
    #
    # This allows the web application to start with the
    # previously indexed documents already available.
    if bm25_storage.exists():
        bm25_index.load()

    # --------------------------------------------------------
    # Vector retrieval
    # --------------------------------------------------------

    # VectorRetriever internally uses:
    #
    #     EmbeddingService
    #            +
    #        QdrantVectorStore
    #
    # These objects are created once and reused.
    vector_retriever = VectorRetriever()

    # --------------------------------------------------------
    # Hybrid retrieval
    # --------------------------------------------------------

    # Combine BM25 and vector retrieval using RRF.
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_index.retriever,
        vector_retriever=vector_retriever,
    )

    # --------------------------------------------------------
    # Application service
    # --------------------------------------------------------

    return RetrievalService(
        retriever=hybrid_retriever,
    )
