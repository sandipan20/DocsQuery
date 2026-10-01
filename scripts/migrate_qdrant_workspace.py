"""
DocsQuery - Qdrant Workspace Migration

Adds workspace_id="public" to the existing DocsQuery corpus.

The original 551 Qdrant points were created before workspace
isolation existed, so they do not have a workspace_id yet.

Those documents belong to the public/shared DocsQuery corpus.
"""

from app.retrieval.vector_store import QdrantVectorStore

BATCH_SIZE = 100


def main() -> None:
    store = QdrantVectorStore()

    print("Qdrant collection:", store.collection_name)

    # --------------------------------------------------------
    # Read existing point IDs from Qdrant.
    #
    # scroll() returns points page-by-page so we do not need
    # to load the entire collection into memory at once.
    # --------------------------------------------------------

    offset = None
    total_updated = 0

    while True:
        points, next_offset = store.client.scroll(
            collection_name=store.collection_name,
            offset=offset,
            limit=BATCH_SIZE,
            with_payload=False,
            with_vectors=False,
        )

        if not points:
            break

        point_ids = [point.id for point in points]

        # ----------------------------------------------------
        # Add workspace ownership to this batch.
        #
        # Existing corpus = public workspace.
        # ----------------------------------------------------

        store.client.set_payload(
            collection_name=store.collection_name,
            payload={
                "workspace_id": "public",
            },
            points=point_ids,
            wait=True,
        )

        total_updated += len(point_ids)

        print(f"Updated {total_updated} points...")

        if next_offset is None:
            break

        offset = next_offset

    # --------------------------------------------------------
    # Verify total number of points.
    # --------------------------------------------------------

    total_points = store.client.count(
        collection_name=store.collection_name,
        exact=True,
    ).count

    print()
    print("Migration complete.")
    print("Points updated:", total_updated)
    print("Total Qdrant points:", total_points)


if __name__ == "__main__":
    main()
