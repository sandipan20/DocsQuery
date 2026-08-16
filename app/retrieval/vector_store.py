"""
DocsQuery - Qdrant Vector Store

This module handles storing and searching document embeddings
inside Qdrant.

Responsibilities:

    DocumentChunk + embedding
            ↓
        Qdrant storage

    Query embedding
            ↓
        Qdrant similarity search
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.config.settings import get_settings
from app.ingestion.models import DocumentChunk


class QdrantVectorStore:
    """
    Wrapper around Qdrant for DocsQuery vector operations.
    """

    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
    ):
        """
        Initialize the Qdrant client.

        Args:
            url:
                Qdrant server URL.

            collection_name:
                Name of the vector collection.
        """

        settings = get_settings()

        self.url = url or settings.qdrant_url

        self.collection_name = (
            collection_name
            or settings.qdrant_collection
        )

        self.client = QdrantClient(
            url=self.url
        )

    def create_collection(
        self,
        vector_size: int,
    ) -> None:
        """
        Create the collection if it doesn't already exist.

        Args:
            vector_size:
                Dimension of the embedding vectors.
        """

        # Check whether the collection already exists.
        collections = self.client.get_collections()

        existing_names = {
            collection.name
            for collection in collections.collections
        }

        if self.collection_name in existing_names:
            return

        # Create a collection using cosine similarity.
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Store document chunks and their embeddings.

        Args:
            chunks:
                Document chunks.

            embeddings:
                Corresponding embedding vectors.

        Raises:
            ValueError:
                If the number of chunks and embeddings differ.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        if not chunks:
            return

        # Make sure the collection exists using the first
        # embedding's dimensionality.
        self.create_collection(
            vector_size=len(embeddings[0])
        )

        points = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            # Qdrant requires a unique point ID.
            #
            # We use the chunk ID because it is already
            # deterministic.
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=embedding,
                    payload={
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "source": chunk.source,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                    },
                )
            )

        # Upsert means:
        #
        # create if new
        # update if already exists
        #
        # This is useful for safe re-ingestion.
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
    ):
        """
        Search Qdrant using a query embedding.

        Args:
            query_vector:
                Embedding of the user's query.

            limit:
                Maximum number of results.

        Returns:
            Qdrant search results.
        """

        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        ).points