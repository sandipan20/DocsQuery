"""
DocsQuery - Hybrid Search CLI

Runs BM25 + vector retrieval and combines both ranked lists
using Reciprocal Rank Fusion (RRF).

Example:

    python -m scripts.search_hybrid "How do I create a new branch?"
"""

import argparse

from app.config.settings import get_settings
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_storage import BM25Storage
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import VectorRetriever


def main() -> None:
    """
    Run hybrid retrieval from the command line.
    """

    parser = argparse.ArgumentParser(
        description="Search DocsQuery using hybrid retrieval."
    )

    # Natural-language query supplied by the user.
    parser.add_argument(
        "query",
        help="Natural-language search query.",
    )

    # Number of final hybrid results to display.
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of final results.",
    )

    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be greater than 0.")

    # --------------------------------------------------------
    # Load application configuration.
    # --------------------------------------------------------

    settings = get_settings()

    # --------------------------------------------------------
    # Restore the persistent BM25 corpus.
    # --------------------------------------------------------

    storage = BM25Storage(settings.bm25_index_path)

    bm25_index = BM25Index(storage=storage)

    bm25_index.load()

    # --------------------------------------------------------
    # Create the vector retriever.
    #
    # This uses:
    #     query -> embedding -> Qdrant
    # --------------------------------------------------------

    vector_retriever = VectorRetriever()

    # --------------------------------------------------------
    # Create the hybrid retriever.
    #
    # It combines:
    #     BM25 ranking
    #     +
    #     Vector ranking
    #     ↓
    #     RRF
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_index.retriever,
        vector_retriever=vector_retriever,
    )

    # Retrieve a larger candidate pool from both systems
    # before selecting the final results.
    candidate_limit = max(
        args.limit * 4,
        20,
    )

    results = hybrid_retriever.retrieve(
        query=args.query,
        limit=args.limit,
        candidate_limit=candidate_limit,
    )

    print()
    print("DocsQuery Hybrid Search")
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
        print(f"RRF Score: {result.score:.6f}")
        print(f"Source: {result.source}")
        print(f"Page: {result.page_number}")
        print(f"Chunk: {result.chunk_id}")
        print()
        print(result.text[:700])
        print("-" * 70)


if __name__ == "__main__":
    main()
