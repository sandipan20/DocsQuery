"""
DocsQuery - Document Ingestion CLI

This script provides a command-line interface for running
the document ingestion pipeline.

Example:

    python -m scripts.ingest_document data/raw/document.pdf

Current pipeline:

    PDF
     ↓
    Stable Document ID
     ↓
    Loader
     ↓
    Cleaner
     ↓
    Chunker
     ↓
    DocumentChunk objects
"""

import argparse

from app.ingestion.pipeline import ingest_pdf


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description="Ingest a PDF into the DocsQuery pipeline."
    )

    # Path to the PDF that should be processed.
    parser.add_argument(
        "file_path",
        help="Path to the PDF document.",
    )

    # Optional chunk size.
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Maximum number of words per chunk.",
    )

    # Optional chunk overlap.
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Number of overlapping words between chunks.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run the document ingestion pipeline and display a summary.
    """

    args = parse_arguments()

    # Run the complete ingestion pipeline.
    chunks = ingest_pdf(
        file_path=args.file_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    # --------------------------------------------------------
    # Display a simple ingestion summary.
    #
    # We don't print the entire document because documents
    # can be extremely large.
    # --------------------------------------------------------

    print()
    print("DocsQuery ingestion completed")
    print("--------------------------------")
    print(f"Source: {args.file_path}")
    print(f"Chunks created: {len(chunks)}")

    if chunks:
        print(f"Document ID: {chunks[0].document_id}")
        print(f"First page: {chunks[0].page_number}")
        print(f"Source file: {chunks[0].source}")

    print("--------------------------------")


if __name__ == "__main__":
    main()
