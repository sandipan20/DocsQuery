"""
DocsQuery - Qdrant Workspace Index

Creates the payload index required for efficient workspace
filtering during vector retrieval.
"""

from qdrant_client.models import PayloadSchemaType

from app.retrieval.vector_store import QdrantVectorStore


def main() -> None:
    store = QdrantVectorStore()

    store.client.create_payload_index(
        collection_name=store.collection_name,
        field_name="workspace_id",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )

    print("workspace_id payload index created.")


if __name__ == "__main__":
    main()
