"""
DocsQuery - Qdrant Migration

Copies the existing local DocsQuery Qdrant collection to Qdrant Cloud.

Source:
    http://localhost:6333

Destination:
    QDRANT_CLOUD_URL
    QDRANT_CLOUD_API_KEY

The migration preserves:
- point IDs
- vectors
- payloads
- collection vector configuration

It does not generate new embeddings.
"""

import os

from qdrant_client import QdrantClient, models

from app.config.settings import get_settings

BATCH_SIZE = 100


def vector_signature(vectors: object) -> object:
    """
    Create a comparable representation of a Qdrant vector configuration.
    """

    if isinstance(vectors, dict):
        return {
            name: (
                params.size,
                str(params.distance),
            )
            for name, params in vectors.items()
        }

    return (
        vectors.size,
        str(vectors.distance),
    )


def main() -> None:
    settings = get_settings()

    cloud_url = os.environ.get("QDRANT_CLOUD_URL", "")
    cloud_api_key = os.environ.get("QDRANT_CLOUD_API_KEY", "")

    if not cloud_url:
        raise RuntimeError("QDRANT_CLOUD_URL is not set.")

    if not cloud_api_key:
        raise RuntimeError("QDRANT_CLOUD_API_KEY is not set.")

    collection_name = settings.qdrant_collection

    print("Connecting to local Qdrant...")
    source = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )

    print("Connecting to Qdrant Cloud...")
    target = QdrantClient(
        url=cloud_url,
        api_key=cloud_api_key,
    )

    # --------------------------------------------------------
    # Inspect source collection
    # --------------------------------------------------------

    if not source.collection_exists(collection_name):
        raise RuntimeError(f"Local collection does not exist: {collection_name}")

    source_info = source.get_collection(collection_name)

    source_count = source.count(
        collection_name=collection_name,
        exact=True,
    ).count

    source_vectors = source_info.config.params.vectors

    print(f"Collection: {collection_name}")
    print(f"Local point count: {source_count}")
    print(f"Vector configuration: {vector_signature(source_vectors)}")

    if source_count == 0:
        raise RuntimeError("Local collection is empty.")

    # --------------------------------------------------------
    # Inspect destination collection
    # --------------------------------------------------------

    if target.collection_exists(collection_name):
        target_count = target.count(
            collection_name=collection_name,
            exact=True,
        ).count

        print("Cloud collection already exists.")
        print(f"Cloud point count: {target_count}")

        if target_count > 0:
            raise RuntimeError(
                "Cloud collection is not empty. "
                "Migration stopped to avoid overwriting existing data."
            )

        target_info = target.get_collection(collection_name)
        target_vectors = target_info.config.params.vectors

        if vector_signature(target_vectors) != vector_signature(source_vectors):
            raise RuntimeError(
                "Cloud collection vector configuration does not match "
                "the local collection."
            )

        print("Existing empty cloud collection is compatible.")

    else:
        print("Creating cloud collection...")

        target.create_collection(
            collection_name=collection_name,
            vectors_config=source_vectors,
        )

        print("Cloud collection created.")

    # --------------------------------------------------------
    # Copy points
    # --------------------------------------------------------

    print("Migrating points...")

    offset = None
    migrated = 0

    while True:
        points, next_offset = source.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=BATCH_SIZE,
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            break

        upload_points = [
            models.PointStruct(
                id=point.id,
                vector=point.vector,
                payload=point.payload,
            )
            for point in points
        ]

        target.upsert(
            collection_name=collection_name,
            points=upload_points,
            wait=True,
        )

        migrated += len(upload_points)

        print(f"Migrated {migrated}/{source_count} points...")

        offset = next_offset

        if offset is None:
            break

    # --------------------------------------------------------
    # Verify migration
    # --------------------------------------------------------

    cloud_count = target.count(
        collection_name=collection_name,
        exact=True,
    ).count

    print()
    print("Migration complete.")
    print(f"Local points : {source_count}")
    print(f"Cloud points : {cloud_count}")

    if cloud_count != source_count:
        raise RuntimeError("Migration verification failed: point counts do not match.")

    print("Qdrant migration verification: OK")


if __name__ == "__main__":
    main()
