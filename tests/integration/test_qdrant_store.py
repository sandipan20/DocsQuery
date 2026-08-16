"""
Integration tests for the real Qdrant vector store.

These tests require a running local Qdrant instance.
"""

from uuid import uuid4

from app.retrieval.vector_store import QdrantVectorStore


def test_qdrant_collection_can_be_created():
    """
    Verify that DocsQuery can connect to Qdrant and create
    a vector collection.
    """

    collection_name = (
        f"test_docsquery_{uuid4().hex}"
    )

    store = QdrantVectorStore(
        collection_name=collection_name,
    )

    store.create_collection(
        vector_size=3,
    )

    collections = (
        store.client.get_collections()
    )

    names = {
        collection.name
        for collection in collections.collections
    }

    assert collection_name in names