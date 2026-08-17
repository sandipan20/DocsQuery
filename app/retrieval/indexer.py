"""
DocsQuery - Vector Indexing Service

Connects the document ingestion pipeline, embedding model,
and Qdrant vector database.

Pipeline:

    PDF
     ↓
    DocumentChunk
     ↓
    EmbeddingService
     ↓
    Embeddings
     ↓
    QdrantVectorStore
     ↓
    Qdrant
"""

from app.ingestion.models import DocumentChunk
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import QdrantVectorStore


class VectorIndexer:
    """
    Index document chunks into Qdrant.

    This class coordinates:
        1. Embedding generation
        2. Vector storage
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: QdrantVectorStore | None = None,
    ):
        """
        Initialize the indexing service.

        Dependencies can be injected, which makes the class
        easier to test.
        """

        self.embedding_service = embedding_service or EmbeddingService()

        self.vector_store = vector_store or QdrantVectorStore()

    def index_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> int:
        """
        Generate embeddings and store document chunks.

        Args:
            chunks:
                Document chunks to index.

        Returns:
            Number of chunks indexed.
        """

        # Nothing to index.
        if not chunks:
            return 0

        # Extract the text that will be embedded.
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings in one batch.
        embeddings = self.embedding_service.embed_texts(texts)

        # Store vectors and metadata in Qdrant.
        self.vector_store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        return len(chunks)
