"""
DocsQuery - Chunk Inspection CLI

This script lets us inspect the chunks stored in the BM25
persistent corpus.

It is useful while building the evaluation dataset because
we need to manually identify which chunks are relevant to
specific questions.

Example:

    python -m scripts.inspect_chunks --query "git branch"
"""

import argparse

from app.config.settings import get_settings
from app.retrieval.bm25_storage import BM25Storage


def main() -> None:
    """
    Find and display chunks containing the requested text.
    """

    # --------------------------------------------------------
    # Parse the command-line argument.
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(description="Inspect stored DocsQuery chunks.")

    parser.add_argument(
        "--query",
        required=True,
        help="Text to search for inside stored chunks.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load the configured BM25 storage path.
    # --------------------------------------------------------

    settings = get_settings()

    storage = BM25Storage(settings.bm25_index_path)

    # --------------------------------------------------------
    # Load all persisted chunks.
    # --------------------------------------------------------

    chunks = storage.load()

    # Perform a simple case-insensitive substring search.
    query = args.query.lower()

    matches = [chunk for chunk in chunks if query in chunk.text.lower()]

    # --------------------------------------------------------
    # Display matching chunks.
    # --------------------------------------------------------

    print()
    print(f"Found {len(matches)} matching chunks.")
    print("=" * 70)

    for chunk in matches:
        print()
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Document ID: {chunk.document_id}")
        print(f"Source: {chunk.source}")
        print(f"Page: {chunk.page_number}")
        print(f"Chunk Index: {chunk.chunk_index}")
        print()
        print(chunk.text[:1500])
        print("-" * 70)


if __name__ == "__main__":
    main()
