"""
DocsQuery - Retrieval Service

Coordinates:

    Vector Confidence Gate
          ↓
    Hybrid Retrieval
          ↓
    Cross-Encoder Reranking

Workspace isolation:

    workspace_id
          ↓
    Vector Retrieval
          ↓
    Confidence Gate
          ↓
    Hybrid Retrieval
          ↓
    Reranking
          ↓
    Workspace-safe Results
"""

from app.retrieval.bm25_index import BM25Index
from app.retrieval.confidence import RetrievalConfidenceGate
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
        Vector Retrieval
          ↓
        Confidence Gate
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
        confidence_gate: RetrievalConfidenceGate,
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

            confidence_gate:
                Semantic confidence gate used to reject
                low-confidence queries before hybrid retrieval.

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
        self.vector_retriever = vector_retriever
        self.reranker = reranker
        self.confidence_gate = confidence_gate

        self.candidate_limit = candidate_limit
        self.top_k = top_k

        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=bm25_index.retriever,
            vector_retriever=vector_retriever,
        )

    def search(
        self,
        query: str,
        limit: int | None = None,
        workspace_id: str = "public",
        document_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """
        Execute the complete retrieval pipeline.

        Args:
            query:
                User's query.

            limit:
                Optional override for final result count.

            workspace_id:
                Workspace whose documents are allowed
                to participate in retrieval.

        Returns:
            Reranked retrieval results belonging only to
            the requested workspace.

        If the vector confidence gate rejects the query,
        an empty result list is returned so the higher-level
        RAG service can handle insufficient evidence.

        Raises:
            ValueError:
                If query is empty, limit is invalid, or
                workspace_id is empty.
        """

        # ----------------------------------------------------
        # Validate inputs.
        # ----------------------------------------------------

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not workspace_id.strip():
            raise ValueError("workspace_id cannot be empty.")

        final_limit = limit if limit is not None else self.top_k

        if final_limit <= 0:
            raise ValueError("limit must be greater than 0.")

        # ----------------------------------------------------
        # Stage 1:
        # Initial vector retrieval.
        #
        # workspace_id is passed to the vector retriever so
        # Qdrant only searches the current workspace.
        #
        # The vector score provides the confidence signal
        # used by the confidence gate.
        # ----------------------------------------------------

        vector_results = self.vector_retriever.retrieve(
            query=query,
            limit=self.candidate_limit,
            workspace_id=workspace_id,
            document_ids=document_ids,
        )

        # ----------------------------------------------------
        # Stage 2:
        # Confidence gate.
        #
        # Low-confidence queries are rejected before the
        # more expensive hybrid retrieval and reranking.
        # ----------------------------------------------------

        if not self.confidence_gate.is_confident(vector_results):
            bm25_results = self.bm25_index.retriever.retrieve(
                query=query,
                limit=self.candidate_limit,
                workspace_id=workspace_id,
                document_ids=document_ids,
            )
            lexical_matches = [result for result in bm25_results if result.score > 0]
            if not lexical_matches:
                if workspace_id != "public":
                    semantic_matches = [
                        result for result in vector_results if result.score >= 0.25
                    ]
                    if semantic_matches:
                        return self.reranker.rerank(
                            query=query,
                            results=semantic_matches,
                            top_k=final_limit,
                        )
                return []
            return self.reranker.rerank(
                query=query,
                results=lexical_matches,
                top_k=final_limit,
            )

        # ----------------------------------------------------
        # Stage 3:
        # Hybrid candidate generation.
        #
        # The already-computed vector results are passed into
        # HybridRetriever so vector retrieval is not repeated.
        #
        # workspace_id is also passed so BM25 retrieval is
        # restricted to the same workspace.
        # ----------------------------------------------------

        candidates = self.hybrid_retriever.retrieve(
            query=query,
            limit=self.candidate_limit,
            candidate_limit=self.candidate_limit,
            vector_results=vector_results,
            workspace_id=workspace_id,
            document_ids=document_ids,
        )

        # ----------------------------------------------------
        # Stage 4:
        # Cross-encoder reranking.
        #
        # Candidates are already workspace-isolated before
        # they reach the reranker.
        # ----------------------------------------------------

        return self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=final_limit,
        )
