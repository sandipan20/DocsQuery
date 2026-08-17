"""
DocsQuery - Vector Search CLI

Example:

    python -m scripts.search "What is Python?"
"""

import argparse

from app.retrieval.vector_retriever import VectorRetriever


def main() -> None:
    """
    Run a vector search from the command line.
    """

    parser = argparse.ArgumentParser(
        description="Search DocsQuery using vector retrieval."
    )

    parser.add_argument(
        "query",
        help="Question or search query.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return.",
    )

    args = parser.parse_args()

    retriever = VectorRetriever()

    results = retriever.retrieve(
        query=args.query,
        limit=args.limit,
    )

    print()
    print("DocsQuery Vector Search")
    print("=" * 60)

    for index, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(f"Result #{index}")
        print(f"Score: {result.score:.4f}")
        print(f"Source: {result.source}")
        print(f"Page: {result.page_number}")
        print(f"Chunk: {result.chunk_id}")
        print()
        print(result.text[:500])
        print("-" * 60)


if __name__ == "__main__":
    main()
