"""
DocsQuery - Hybrid Retriever

Combines:
    1. BM25 keyword retrieval
    2. Vector semantic retrieval

using Reciprocal Rank Fusion (RRF).

Pipeline:

    User Query
        │
        ├───────────────┐
        ▼               ▼
      BM25           Vector
        │               │
        ▼               ▼
     Results         Results
        │               │
        └───────┬───────┘
                ▼
              RRF
                │
                ▼
        Hybrid Results

Workspace isolation:

    User Session
        │
        ▼
    workspace_id
        │
        ├───────────────┐
        ▼               ▼
      BM25           Vector
        │               │
        ▼               ▼
  Only matching    Only matching
   workspace        workspace
        │               │
        └───────┬───────┘
                ▼
              RRF
"""

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.models import RetrievalResult
from app.retrieval.vector_retriever import VectorRetriever


class HybridRetriever:
    """
    Combines BM25 and vector retrieval using RRF.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        rrf_k: int = 60,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            bm25_retriever:
                Keyword-based retriever.

            vector_retriever:
                Semantic/vector retriever.

            rrf_k:
                RRF constant used to reduce the influence
                of very high rankings.
        """

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0.")

        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        candidate_limit: int = 20,
        vector_results: list[RetrievalResult] | None = None,
        workspace_id: str = "public",
        document_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve and fuse results from BM25 and vector search.

        Args:
            query:
                User's search query.

            limit:
                Number of final results.

            candidate_limit:
                Number of candidates retrieved from each
                retrieval system before fusion.

            vector_results:
                Optional precomputed vector results.

                When supplied, these results are used directly.
                Otherwise vector retrieval is performed using
                the requested workspace_id.

            workspace_id:
                Workspace whose documents are allowed to
                participate in retrieval.

        Returns:
            RRF-ranked hybrid results belonging to the
            requested workspace.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be greater than 0.")

        if not workspace_id.strip():
            raise ValueError("workspace_id cannot be empty.")

        # ----------------------------------------------------
        # Retrieve candidates from BM25.
        #
        # BM25Retriever performs the workspace filtering,
        # so chunks from another workspace never enter RRF.
        # ----------------------------------------------------

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            limit=candidate_limit,
            workspace_id=workspace_id,
            document_ids=document_ids,
        )

        # ----------------------------------------------------
        # Retrieve candidates from vector search.
        #
        # Qdrant filtering is handled inside VectorRetriever
        # / QdrantVectorStore using workspace_id.
        # ----------------------------------------------------

        if vector_results is None:
            vector_results = self.vector_retriever.retrieve(
                query=query,
                limit=candidate_limit,
                workspace_id=workspace_id,
                document_ids=document_ids,
            )

        # ----------------------------------------------------
        # Fuse the ranked lists using RRF.
        # ----------------------------------------------------

        scores: dict[str, float] = {}

        results_by_id: dict[str, RetrievalResult] = {}

        self._add_ranked_results(
            results=bm25_results,
            scores=scores,
            results_by_id=results_by_id,
        )

        self._add_ranked_results(
            results=vector_results,
            scores=scores,
            results_by_id=results_by_id,
        )

        # ----------------------------------------------------
        # Sort by fused RRF score.
        # ----------------------------------------------------

        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        final_results: list[RetrievalResult] = []

        for chunk_id in ranked_ids[:limit]:
            result = results_by_id[chunk_id]

            # Replace the individual retriever score with
            # the final hybrid RRF score.
            final_results.append(
                result.model_copy(
                    update={
                        "score": scores[chunk_id],
                    }
                )
            )

        return final_results

    def _add_ranked_results(
        self,
        results: list[RetrievalResult],
        scores: dict[str, float],
        results_by_id: dict[str, RetrievalResult],
    ) -> None:
        """
        Add one ranked result list to the RRF accumulator.

        RRF formula:

            1 / (rrf_k + rank)

        A result appearing in both retrieval systems receives
        contributions from both rankings.
        """

        for rank, result in enumerate(
            results,
            start=1,
        ):
            chunk_id = result.chunk_id

            # RRF contribution:
            #
            # 1 / (k + rank)
            contribution = 1.0 / (self.rrf_k + rank)

            scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution

            # Keep one copy of the result metadata.
            results_by_id.setdefault(
                chunk_id,
                result,
            )
