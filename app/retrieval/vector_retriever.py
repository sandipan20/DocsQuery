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

Workspace isolation:

    User Query
        ↓
    workspace_id
        ↓
    Embedding
        ↓
    Qdrant
        ↓
    Only chunks belonging to that workspace
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

        Args:
            embedding_service:
                Service responsible for converting text
                into embedding vectors.

            vector_store:
                Qdrant vector store used for semantic search.
        """

        self.embedding_service = embedding_service or EmbeddingService()

        self.vector_store = vector_store or QdrantVectorStore()

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        workspace_id: str = "public",
        document_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve documents semantically similar to a query.

        Args:
            query:
                User's natural-language question.

            limit:
                Maximum number of results.

            workspace_id:
                Workspace whose documents are allowed
                to appear in the results.

        Returns:
            Ranked RetrievalResult objects belonging only
            to the requested workspace.

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

        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        if not workspace_id.strip():
            raise ValueError("workspace_id cannot be empty.")

        # ----------------------------------------------------
        # Convert the user's query into the same vector space
        # used when indexing document chunks.
        # ----------------------------------------------------

        query_vector = self.embedding_service.embed_text(query)

        # ----------------------------------------------------
        # Search Qdrant.
        #
        # workspace_id is passed to the vector store so that
        # Qdrant applies the workspace payload filter during
        # retrieval.
        #
        # This is important for security because filtering
        # after retrieval would allow another workspace's
        # chunks to enter the retrieval pipeline.
        # ----------------------------------------------------

        return self.vector_store.search(
            query_vector=query_vector,
            limit=limit,
            workspace_id=workspace_id,
            document_ids=document_ids,
        )
