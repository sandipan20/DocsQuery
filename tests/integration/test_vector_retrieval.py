"""Integration test for real vector retrieval against an isolated fixture."""

from uuid import uuid4

from app.ingestion.models import DocumentChunk
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.vector_store import QdrantVectorStore


def test_vector_retrieval_returns_results():
    """
    Verify that a real query can retrieve indexed chunks.
    """

    collection_name = f"test_vector_retrieval_{uuid4().hex}"
    vector_store = QdrantVectorStore(collection_name=collection_name)
    retriever = VectorRetriever(vector_store=vector_store)
    chunk = DocumentChunk(
        chunk_id="integration-python-chunk",
        document_id="integration-python-document",
        workspace_id="integration-test",
        text="Python is a programming language used to build many kinds of software.",
        source="integration-fixture.pdf",
        page_number=1,
        chunk_index=0,
    )

    try:
        vector_store.upsert_chunks(
            chunks=[chunk],
            embeddings=retriever.embedding_service.embed_texts([chunk.text]),
        )
        results = retriever.retrieve(
            "What is Python?",
            limit=3,
            workspace_id="integration-test",
        )
    finally:
        vector_store.client.delete_collection(collection_name=collection_name)

    assert len(results) > 0

    # Results should be ranked with numerical scores.
    assert all(isinstance(result.score, float) for result in results)

    # Every result must have citation metadata.
    assert all(result.source and result.page_number >= 1 for result in results)
    assert all(result.workspace_id == "integration-test" for result in results)
