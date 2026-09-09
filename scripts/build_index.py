"""
DocsQuery - Canonical Retrieval Index Builder

Builds the complete retrieval state from the source PDF corpus.

Pipeline:

    data/raw/*.pdf
          ↓
    corpus manifest
          ↓
    PDF ingestion
          ↓
    DocumentChunk objects
          ↓
    RetrievalIndexManager.rebuild()
          ├── BM25
          └── Qdrant
          ↓
    index metadata

This is the command that should be used whenever the source corpus
changes and the retrieval indexes need to be rebuilt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config.settings import get_settings
from app.ingestion.pipeline import ingest_pdf
from app.retrieval.index_manager import RetrievalIndexManager
from app.storage.corpus_manifest import (
    build_corpus_manifest,
    save_manifest,
)
from app.storage.index_metadata import (
    create_index_metadata,
    save_index_metadata,
    validate_index_metadata,
)

DEFAULT_CORPUS_ROOT = "data/raw"

DEFAULT_MANIFEST_PATH = "data/index/corpus_manifest.json"

DEFAULT_METADATA_PATH = "data/index/index_metadata.json"

DEFAULT_CHUNK_SIZE = 500

DEFAULT_CHUNK_OVERLAP = 50

DEFAULT_DATASET_VERSION = "1.1.0"

DEFAULT_RRF_K = 60


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Build DocsQuery retrieval indexes.")

    parser.add_argument(
        "--corpus-root",
        default=DEFAULT_CORPUS_ROOT,
        help="Directory containing source PDF files.",
    )

    parser.add_argument(
        "--manifest-path",
        default=DEFAULT_MANIFEST_PATH,
        help="Where the corpus manifest should be written.",
    )

    parser.add_argument(
        "--metadata-path",
        default=DEFAULT_METADATA_PATH,
        help="Where index build metadata should be written.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum words per chunk.",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Words shared between adjacent chunks.",
    )

    parser.add_argument(
        "--dataset-version",
        default=DEFAULT_DATASET_VERSION,
        help="Evaluation dataset version associated with this build.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF constant recorded with the index build.",
    )

    return parser.parse_args()


def build_index(
    *,
    corpus_root: str,
    manifest_path: str,
    metadata_path: str,
    chunk_size: int,
    chunk_overlap: int,
    dataset_version: str,
    rrf_k: int,
) -> int:
    """
    Build BM25 and Qdrant from the complete corpus.

    Returns:
        Number of indexed chunks.
    """

    settings = get_settings()

    # ------------------------------------------------------------
    # Validate the chunk configuration before doing expensive work.
    # ------------------------------------------------------------

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero.")

    corpus_path = Path(corpus_root)

    print()
    print("=" * 70)
    print("DocsQuery Index Build")
    print("=" * 70)

    # ------------------------------------------------------------
    # Stage 1:
    # Fingerprint the source corpus.
    # ------------------------------------------------------------

    manifest = build_corpus_manifest(corpus_path)

    save_manifest(
        manifest,
        manifest_path,
    )

    print(f"Corpus files:       {len(manifest.files)}")

    print(f"Corpus SHA-256:     {manifest.corpus_sha256}")

    # ------------------------------------------------------------
    # Stage 2:
    # Ingest documents in deterministic order.
    # ------------------------------------------------------------

    all_chunks = []

    for corpus_file in manifest.files:
        pdf_path = corpus_path / corpus_file.path

        print()
        print(f"Ingesting: {pdf_path}")

        chunks = ingest_pdf(
            file_path=str(pdf_path),
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )

        print(f"  chunks: {len(chunks)}")

        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("Corpus ingestion produced zero chunks.")

    # ------------------------------------------------------------
    # Stage 3:
    # Protect the global chunk-ID invariant.
    # ------------------------------------------------------------

    chunk_ids = [chunk.chunk_id for chunk in all_chunks]

    unique_chunk_ids = set(chunk_ids)

    if len(chunk_ids) != len(unique_chunk_ids):
        raise RuntimeError("Duplicate chunk IDs detected before indexing.")

    document_ids = {chunk.document_id for chunk in all_chunks}

    print()
    print(f"Documents:          {len(document_ids)}")

    print(f"Chunks:             {len(all_chunks)}")

    # ------------------------------------------------------------
    # Stage 4:
    # Rebuild both retrieval indexes from exactly the same chunks.
    # ------------------------------------------------------------

    print()
    print("Rebuilding BM25 and Qdrant...")

    index_manager = RetrievalIndexManager()

    indexed_count = index_manager.rebuild(all_chunks)

    if indexed_count != len(all_chunks):
        raise RuntimeError(
            f"Index count mismatch: expected {len(all_chunks)}, got {indexed_count}."
        )

    # ------------------------------------------------------------
    # Stage 5:
    # Record the exact configuration of this build.
    # ------------------------------------------------------------

    metadata = create_index_metadata(
        corpus_sha256=manifest.corpus_sha256,
        corpus_files=manifest.files,
        dataset_version=dataset_version,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        rrf_k=rrf_k,
        qdrant_collection=settings.qdrant_collection,
        document_count=len(document_ids),
        chunk_count=len(all_chunks),
    )

    validate_index_metadata(metadata)

    save_index_metadata(
        metadata,
        metadata_path,
    )

    print()
    print(f"Manifest:           {manifest_path}")

    print(f"Index metadata:     {metadata_path}")

    print(f"Indexed chunks:     {indexed_count}")

    print()
    print("=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)

    return indexed_count


def main() -> None:
    """CLI entry point."""

    args = parse_args()

    build_index(
        corpus_root=args.corpus_root,
        manifest_path=args.manifest_path,
        metadata_path=args.metadata_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        dataset_version=args.dataset_version,
        rrf_k=args.rrf_k,
    )


if __name__ == "__main__":
    main()
