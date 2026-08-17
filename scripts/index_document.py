"""
DocsQuery - Document Indexing CLI

Indexes a PDF into Qdrant.

Pipeline:

    PDF
     ↓
    Ingestion
     ↓
    DocumentChunk
     ↓
    Embedding
     ↓
    Qdrant
"""

import argparse

from app.ingestion.pipeline import ingest_pdf
from app.retrieval.indexer import VectorIndexer


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Index a PDF into DocsQuery."
    )

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
    """

    args = parse_arguments()

    print()
    print("Starting DocsQuery indexing...")
    print("--------------------------------")

    # Step 1:
    # Convert the PDF into searchable document chunks.
    chunks = ingest_pdf(
        file_path=args.file_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print(f"Chunks created: {len(chunks)}")

    # Step 2:
    # Generate embeddings and store everything in Qdrant.
    indexer = VectorIndexer()

    indexed_count = indexer.index_chunks(chunks)

    print(f"Chunks indexed: {indexed_count}")

    if chunks:
        print(f"Document ID: {chunks[0].document_id}")

    print("--------------------------------")
    print("Indexing completed successfully.")


if __name__ == "__main__":
    main()