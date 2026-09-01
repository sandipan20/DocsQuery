"""
DocsQuery - BM25 Search CLI

Runs keyword-based BM25 retrieval against the persisted
document corpus.

Example:

    python -m scripts.search_bm25 "git branch"
"""

import argparse

from app.config.settings import get_settings
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_storage import BM25Storage


def main() -> None:
    """
    Run a BM25 search from the command line.
    """

    parser = argparse.ArgumentParser(description="Search DocsQuery using BM25.")

    # The search query supplied by the user.
    parser.add_argument(
        "query",
        help="Keyword search query.",
    )

    # Maximum number of results to display.
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return.",
    )

    args = parser.parse_args()

    # Load application configuration.
    settings = get_settings()

    # Open the persistent BM25 corpus.
    storage = BM25Storage(settings.bm25_index_path)

    # Create the BM25 index manager.
    index = BM25Index(storage=storage)

    # Restore the persisted chunks and rebuild the
    # searchable BM25 structure in memory.
    index.load()

    # Execute the keyword search.
    results = index.search(
        query=args.query,
        limit=args.limit,
    )

    print()
    print("DocsQuery BM25 Search")
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    for position, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(f"Result #{position}")
        print(f"BM25 Score: {result.score:.4f}")
        print(f"Source: {result.source}")
        print(f"Page: {result.page_number}")
        print(f"Chunk: {result.chunk_id}")
        print()
        print(result.text[:700])
        print("-" * 70)


if __name__ == "__main__":
    main()
