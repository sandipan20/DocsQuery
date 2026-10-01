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

from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.config.settings import get_settings
from app.ingestion.models import DocumentChunk
from app.retrieval.models import RetrievalResult


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

        self.collection_name = collection_name or settings.qdrant_collection

        # Connect to Qdrant.
        #
        # For local development this can connect to localhost.
        # For production/cloud this uses the configured API key.
        self.client = QdrantClient(
            url=self.url,
            api_key=settings.qdrant_api_key or None,
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

        existing_names = {collection.name for collection in collections.collections}

        # Don't recreate an existing collection.
        if self.collection_name in existing_names:
            self._ensure_payload_indexes()
            return

        # Create a collection using cosine similarity.
        #
        # Our embedding service generates normalized vectors,
        # so cosine similarity is appropriate for semantic search.
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        for field_name in ("workspace_id", "document_id"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
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

        # Every chunk must have exactly one embedding.
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings.")

        # Nothing to store.
        if not chunks:
            return

        # Create the collection if necessary.
        #
        # The first embedding tells us the vector dimension.
        self.create_collection(
            vector_size=len(embeddings[0]),
        )

        points = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            # ------------------------------------------------
            # SECURITY / ISOLATION:
            #
            # Include workspace_id in the deterministic point ID.
            #
            # Why?
            # ----
            # Two different users can upload the exact same PDF.
            #
            # Because document_id and chunk_id are deterministic,
            # the chunk IDs can otherwise be identical.
            #
            # Including workspace_id prevents one user's point
            # from overwriting another user's point.
            # ------------------------------------------------

            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{chunk.workspace_id}:{chunk.chunk_id}",
                )
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        # Application-level document identity.
                        "document_id": chunk.document_id,
                        # Workspace that owns this chunk.
                        "workspace_id": chunk.workspace_id,
                        # Application-level chunk identity.
                        "chunk_id": chunk.chunk_id,
                        # Actual text used during retrieval.
                        "text": chunk.text,
                        # Citation metadata.
                        "source": chunk.source,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "document_page_count": chunk.document_page_count,
                    },
                )
            )

        # ----------------------------------------------------
        # Upsert means:
        #
        #   new point      → create
        #   existing point → update
        #
        # Because the point ID is deterministic within a
        # workspace, re-indexing the same document updates
        # the same point instead of creating duplicates.
        # ----------------------------------------------------

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        workspace_id: str = "public",
        document_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """
        Search Qdrant using a query embedding.

        Args:
            query_vector:
                Embedding of the user's query.

            limit:
                Maximum number of results.

            workspace_id:
                Workspace whose documents are allowed to
                participate in retrieval.

        Returns:
            RetrievalResult objects ordered by relevance.
        """

        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        if not workspace_id.strip():
            raise ValueError("workspace_id cannot be empty.")
        if document_ids == []:
            return []

        # ----------------------------------------------------
        # SECURITY:
        #
        # Apply the workspace restriction INSIDE Qdrant.
        #
        # We do NOT:
        #
        #   1. retrieve all users' documents
        #   2. retrieve top results
        #   3. filter them afterward
        #
        # Qdrant performs the ownership filter as part of
        # the vector query itself.
        # ----------------------------------------------------

        workspace_filter = self._document_filter(
            workspace_id=workspace_id,
            document_ids=document_ids,
        )

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=workspace_filter,
                limit=limit,
                with_payload=True,
            ).points
        except Exception as exc:
            if "not found" in str(exc).lower():
                return []
            raise

        retrieval_results = []

        for result in results:
            payload = result.payload or {}

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=payload["chunk_id"],
                    document_id=payload["document_id"],
                    workspace_id=payload.get(
                        "workspace_id",
                        "public",
                    ),
                    text=payload["text"],
                    source=payload["source"],
                    page_number=payload["page_number"],
                    chunk_index=payload["chunk_index"],
                    score=float(result.score),
                )
            )

        return retrieval_results

    def scroll_document_chunks(
        self,
        workspace_id: str,
        document_id: str | None = None,
    ) -> list[dict]:
        if not workspace_id.strip():
            raise ValueError("workspace_id cannot be empty.")

        query_filter = self._document_filter(
            workspace_id=workspace_id,
            document_ids=[document_id] if document_id else None,
        )
        chunks: list[dict] = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(point.payload or {} for point in points)
            if offset is None:
                return chunks

    def delete_document(
        self,
        workspace_id: str,
        document_id: str,
    ) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=self._document_filter(
                    workspace_id=workspace_id,
                    document_ids=[document_id],
                )
            ),
        )

    @staticmethod
    def _document_filter(
        workspace_id: str,
        document_ids: list[str] | None = None,
    ) -> Filter:
        if not workspace_id.strip():
            raise ValueError("workspace_id cannot be empty.")
        if document_ids == []:
            raise ValueError("document_ids cannot be empty.")
        conditions = [
            FieldCondition(
                key="workspace_id",
                match=MatchValue(value=workspace_id),
            )
        ]
        if document_ids:
            conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_ids[0])
                    if len(document_ids) == 1
                    else MatchAny(any=document_ids),
                )
            )
        return Filter(must=conditions)

    def recreate_collection(
        self,
        vector_size: int,
    ) -> None:
        """
        Delete the existing Qdrant collection and create
        a completely fresh one.

        This is used when rebuilding the entire corpus from
        scratch during development and evaluation.

        WARNING:
            This permanently deletes the current collection
            and all vectors stored inside it.
        """

        collections = self.client.get_collections()

        existing_names = {collection.name for collection in collections.collections}

        # Delete the old collection when it exists.
        if self.collection_name in existing_names:
            self.client.delete_collection(
                collection_name=self.collection_name,
            )

        # Create a completely clean collection.
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
