"""
DocsQuery - Retrieval Service

Application-level service that coordinates retrieval.

The API layer should not know how BM25, Qdrant, embeddings,
or RRF work internally.

Architecture:

    FastAPI
       ↓
    RetrievalService
       ↓
    HybridRetriever
       ├── BM25
       └── Vector/Qdrant
       ↓
    RetrievalResult
"""

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import RetrievalResult


class RetrievalService:
    """
    Application service for document retrieval.

    The service depends on a retriever interface rather than
    constructing the retrieval system itself.

    This makes the service:
        - easy to test
        - easy to replace
        - independent of Qdrant/BM25 implementation details
    """

    def __init__(
        self,
        retriever: HybridRetriever,
    ):
        """
        Initialize the retrieval service.

        Args:
            retriever:
                Configured hybrid retriever.
        """

        self.retriever = retriever

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        """
        Execute document retrieval.

        Args:
            query:
                User's search query.

            limit:
                Number of final results.

        Returns:
            Ranked retrieval results.

        Raises:
            ValueError:
                If the retriever rejects the query or limit.
        """

        # The service delegates retrieval to the configured
        # retriever. The retriever itself owns its internal
        # candidate-selection strategy.
        return self.retriever.retrieve(
            query=query,
            limit=limit,
        )
