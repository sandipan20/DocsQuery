"""
DocsQuery - Retrieval Service

Coordinates:

    Hybrid Retrieval
          ↓
    Cross-Encoder Reranking
"""

from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import RetrievalResult
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_retriever import VectorRetriever


class RetrievalService:
    """
    Application-level retrieval service.

    Retrieval pipeline:

        Query
          ↓
        BM25 + Vector
          ↓
        RRF
          ↓
        Candidate Results
          ↓
        Cross Encoder
          ↓
        Final Results
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        vector_retriever: VectorRetriever,
        reranker: CrossEncoderReranker,
        candidate_limit: int = 20,
        top_k: int = 5,
    ):
        """
        Initialize the retrieval service.

        Args:
            bm25_index:
                Persistent BM25 index.

            vector_retriever:
                Qdrant-backed vector retriever.

            reranker:
                Cross-encoder reranker.

            candidate_limit:
                Number of hybrid candidates sent to the
                reranker.

            top_k:
                Number of final results returned.
        """

        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be greater than 0.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        self.bm25_index = bm25_index

        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=bm25_index.retriever,
            vector_retriever=vector_retriever,
        )

        self.reranker = reranker

        self.candidate_limit = candidate_limit
        self.top_k = top_k

    def search(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Execute the complete retrieval pipeline.

        Args:
            query:
                User's query.

            limit:
                Optional override for final result count.

        Returns:
            Reranked retrieval results.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        final_limit = limit if limit is not None else self.top_k

        if final_limit <= 0:
            raise ValueError("limit must be greater than 0.")

        # ----------------------------------------------------
        # Stage 1:
        # Hybrid candidate generation.
        # ----------------------------------------------------

        candidates = self.hybrid_retriever.retrieve(
            query=query,
            limit=self.candidate_limit,
            candidate_limit=self.candidate_limit,
        )

        # ----------------------------------------------------
        # Stage 2:
        # Cross-encoder reranking.
        # ----------------------------------------------------

        return self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=final_limit,
        )
