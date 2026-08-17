"""
DocsQuery - Document Indexing CLI

Indexes a PDF into all retrieval systems.

Pipeline:

    PDF
     ↓
    Ingestion
     ↓
    DocumentChunk
     ↓
    ┌───────────────────────┐
    │ RetrievalIndexManager │
    └───────────┬───────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
      BM25             Vector
        │                │
        ▼                ▼
   bm25.json           Qdrant
"""

import argparse

from app.ingestion.pipeline import ingest_pdf
from app.retrieval.index_manager import RetrievalIndexManager


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Index a PDF into DocsQuery.")

    parser.add_argument(
        "file_path",
        help="Path to the PDF document.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Maximum words per chunk.",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Number of overlapping words.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run the complete PDF indexing workflow.

    Both retrieval systems are updated:

        1. BM25
        2. Qdrant vector search
    """

    args = parse_arguments()

    print()
    print("Starting DocsQuery indexing...")
    print("--------------------------------")

    # --------------------------------------------------------
    # Step 1:
    # Convert the PDF into searchable document chunks.
    # --------------------------------------------------------

    chunks = ingest_pdf(
        file_path=args.file_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print(f"Chunks created: {len(chunks)}")

    # --------------------------------------------------------
    # Step 2:
    # Index the same chunks into BOTH retrieval systems.
    #
    # BM25:
    #     data/index/bm25.json
    #
    # Vector:
    #     Qdrant
    # --------------------------------------------------------

    index_manager = RetrievalIndexManager()

    indexed_count = index_manager.index(chunks)

    print(f"Chunks indexed: {indexed_count}")

    # --------------------------------------------------------
    # Step 3:
    # Display document information.
    # --------------------------------------------------------

    if chunks:
        print(f"Document ID: {chunks[0].document_id}")

    print("--------------------------------")
    print("Indexing completed successfully.")


if __name__ == "__main__":
    main()
