"""
DocsQuery - Benchmark Candidate Explorer

Searches the indexed corpus and prints candidate chunks that
can be reviewed when constructing retrieval benchmark examples.

This tool does not modify the benchmark dataset.
"""

import argparse

from app.config.settings import get_settings
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_storage import BM25Storage


def main() -> None:
    """
    Search the BM25 corpus for benchmark-authoring candidates.
    """

    parser = argparse.ArgumentParser(
        description="Find corpus chunks for retrieval benchmark construction."
    )

    parser.add_argument(
        "query",
        help="Concept or question to search for.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of candidate chunks to display.",
    )

    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be greater than 0.")

    settings = get_settings()

    index = BM25Index(
        storage=BM25Storage(
            settings.bm25_index_path,
        ),
    )

    index.load()

    results = index.search(
        query=args.query,
        limit=args.limit,
    )

    print()
    print("DocsQuery Benchmark Candidate Explorer")
    print("=" * 100)
    print(f"Query: {args.query}")
    print(f"Candidates: {len(results)}")
    print("=" * 100)

    for rank, result in enumerate(results, start=1):
        print()
        print("-" * 100)
        print(f"Rank:       {rank}")
        print(f"Score:      {result.score:.6f}")
        print(f"Chunk:      {result.chunk_id}")
        print(f"Source:     {result.source}")
        print(f"Page:       {result.page_number}")
        print("-" * 100)
        print(result.text)


if __name__ == "__main__":
    main()
