"""
DocsQuery - Corpus Ingestion CLI

Ingests every PDF inside a directory.

Example:

    python -m scripts.ingest_corpus data/raw

The script:

    PDF files
       ↓
    ingestion pipeline
       ↓
    DocumentChunk objects
       ↓
    persistent BM25 index
       +
    Qdrant vector index

This gives us a reproducible way to rebuild the complete
retrieval corpus whenever the source documents change.
"""

import argparse
from pathlib import Path

from app.config.settings import get_settings
from app.ingestion.pipeline import ingest_pdf
from app.retrieval.index_manager import RetrievalIndexManager


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Ingest a directory of PDF documents.")

    parser.add_argument(
        "directory",
        help="Directory containing PDF documents.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Maximum number of words per chunk.",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Number of overlapping words.",
    )

    return parser.parse_args()


def find_pdfs(directory: Path) -> list[Path]:
    """
    Find all PDF files in a directory.

    Files are sorted to make ingestion deterministic.
    """

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def main() -> None:
    """
    Ingest the complete PDF corpus.
    """

    args = parse_arguments()

    directory = Path(args.directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Expected a directory: {directory}")

    pdfs = find_pdfs(directory)

    if not pdfs:
        raise ValueError(f"No PDF files found in: {directory}")

    print()
    print("DocsQuery Corpus Ingestion")
    print("=" * 70)
    print(f"Documents found: {len(pdfs)}")

    all_chunks = []

    for pdf_path in pdfs:
        print()
        print(f"Ingesting: {pdf_path.name}")

        chunks = ingest_pdf(
            file_path=str(pdf_path),
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

        print(f"  Chunks created: {len(chunks)}")

        if chunks:
            print(f"  Document ID: {chunks[0].document_id}")

        all_chunks.extend(chunks)

    print()
    print(f"Total chunks created: {len(all_chunks)}")

    # Create the retrieval index manager.
    index_manager = RetrievalIndexManager()

    # Build both BM25 and vector indexes from the same corpus.
    indexed_count = index_manager.rebuild(all_chunks)

    print(f"Total chunks indexed: {indexed_count}")

    settings = get_settings()

    print()
    print(f"BM25 corpus: {settings.bm25_index_path}")

    print(f"Qdrant collection: {settings.qdrant_collection}")

    print("=" * 70)
    print("Corpus ingestion completed successfully.")


if __name__ == "__main__":
    main()
