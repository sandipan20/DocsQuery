"""
DocsQuery - Vector Retriever

Converts a user's query into an embedding and searches Qdrant.

Pipeline:

    User Query
        ↓
    EmbeddingService
        ↓
    Query Vector
        ↓
    Qdrant
        ↓
    RetrievalResult
"""

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.models import RetrievalResult
from app.retrieval.vector_store import QdrantVectorStore


class VectorRetriever:
    """
    Semantic/vector-based document retriever.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: QdrantVectorStore | None = None,
    ):
        """
        Initialize the vector retriever.

        Dependencies are injectable so the retriever can be
        tested without loading a real embedding model or
        connecting to Qdrant.
        """

        self.embedding_service = embedding_service or EmbeddingService()

        self.vector_store = vector_store or QdrantVectorStore()

    def retrieve(
        self,
        query: str,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """
        Retrieve documents semantically similar to a query.

        Args:
            query:
                User's natural-language question.

            limit:
                Maximum number of results.

        Returns:
            Ranked RetrievalResult objects.

        Raises:
            ValueError:
                If query is empty or limit is invalid.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        # Convert the user's query into the same vector space
        # used when indexing documents.
        query_vector = self.embedding_service.embed_text(query)

        # Search Qdrant using the query vector.
        return self.vector_store.search(
            query_vector=query_vector,
            limit=limit,
        )
