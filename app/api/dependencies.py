"""
DocsQuery - API Dependencies

Provides application-level dependencies to FastAPI routes.

The retrieval service contains:

    BM25
    Vector Retrieval
    Qdrant
    Hybrid RRF
    Cross-Encoder Reranking
    Retrieval Confidence Gate

The retrieval service is created once and cached so that expensive
components such as embedding and reranking models are not recreated
for every HTTP request.
"""

from functools import lru_cache

from app.config.settings import get_settings
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_storage import BM25Storage
from app.retrieval.confidence import RetrievalConfidenceGate
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_retriever import VectorRetriever
from app.services.retrieval_service import RetrievalService


@lru_cache
def get_retrieval_service() -> RetrievalService:
    """
    Create and cache the application's retrieval service.

    The first call creates the complete retrieval stack.

    Later calls return the same instance.

    This prevents expensive objects such as embedding and
    reranking models from being recreated for every request.
    """

    settings = get_settings()

    # --------------------------------------------------------
    # Retrieval confidence gate
    # --------------------------------------------------------

    confidence_gate = RetrievalConfidenceGate(
        vector_threshold=settings.vector_confidence_threshold,
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_storage = BM25Storage(
        settings.bm25_index_path,
    )

    bm25_index = BM25Index(
        storage=bm25_storage,
    )

    # Load the persisted BM25 corpus when it exists.
    if bm25_storage.exists():
        bm25_index.load()

    # --------------------------------------------------------
    # Vector retrieval
    # --------------------------------------------------------

    vector_retriever = VectorRetriever()

    # --------------------------------------------------------
    # Cross-encoder reranker
    # --------------------------------------------------------

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
    )

    # --------------------------------------------------------
    # Complete application retrieval service
    # --------------------------------------------------------

    return RetrievalService(
        bm25_index=bm25_index,
        vector_retriever=vector_retriever,
        reranker=reranker,
        confidence_gate=confidence_gate,
        candidate_limit=settings.top_k_dense,
        top_k=settings.reranker_top_k,
    )
