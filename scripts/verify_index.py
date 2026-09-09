"""
DocsQuery - Retrieval Index Validator

Verifies that the current corpus and retrieval index metadata
belong together.

Validation flow:

    Current corpus
          ↓
    Corpus manifest
          ↓
    Index metadata
          ↓
    Application configuration
          ↓
    Validation result

The validator does not rebuild the index. It only checks whether
the existing index metadata is consistent with the current source
corpus and configuration.
"""

import argparse
from pathlib import Path

from app.config.settings import get_settings
from app.storage.corpus_manifest import (
    build_corpus_manifest,
    load_manifest,
)
from app.storage.index_metadata import (
    load_index_metadata,
    validate_index_metadata,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Validate DocsQuery corpus and index metadata."
    )

    parser.add_argument(
        "--corpus-root",
        default="data/raw",
        help="Root directory containing source PDFs.",
    )

    parser.add_argument(
        "--manifest-path",
        default="data/index/corpus_manifest.json",
        help="Path to the corpus manifest.",
    )

    parser.add_argument(
        "--metadata-path",
        default="data/index/index_metadata.json",
        help="Path to index metadata.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Expected chunk size.",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Expected chunk overlap.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Expected Reciprocal Rank Fusion constant.",
    )

    return parser.parse_args()


def verify_index(
    *,
    corpus_root: str,
    manifest_path: str,
    metadata_path: str,
    chunk_size: int,
    chunk_overlap: int,
    rrf_k: int,
) -> None:
    """
    Validate the corpus, manifest, index metadata, and
    application configuration.

    Raises:
        RuntimeError:
            If any consistency check fails.
    """

    settings = get_settings()

    corpus_path = Path(corpus_root)

    manifest = load_manifest(manifest_path)

    metadata = load_index_metadata(metadata_path)

    # --------------------------------------------------------
    # Stage 1:
    # Validate the metadata structure itself.
    # --------------------------------------------------------

    validate_index_metadata(metadata)

    # --------------------------------------------------------
    # Stage 2:
    # Rebuild the corpus fingerprint from the current files.
    # --------------------------------------------------------

    current_manifest = build_corpus_manifest(corpus_path)

    if current_manifest.corpus_sha256 != manifest.corpus_sha256:
        raise RuntimeError("Current corpus does not match the stored corpus manifest.")

    # --------------------------------------------------------
    # Stage 3:
    # Verify that the index was built from the same corpus.
    # --------------------------------------------------------

    if metadata.corpus_sha256 != manifest.corpus_sha256:
        raise RuntimeError(
            "Index metadata corpus hash does not match the stored corpus manifest."
        )

    # Index metadata stores corpus files as a tuple, while the
    # manifest stores them as a list. Normalize both to tuples
    # before comparing their actual CorpusFile contents.
    if tuple(metadata.corpus_files) != tuple(manifest.files):
        raise RuntimeError(
            "Index metadata corpus files do not match the stored corpus manifest."
        )

    # --------------------------------------------------------
    # Stage 4:
    # Verify document count.
    # --------------------------------------------------------

    expected_document_count = len(manifest.files)

    if metadata.document_count != expected_document_count:
        raise RuntimeError(
            "Index metadata document count does not match "
            f"the corpus: expected {expected_document_count}, "
            f"got {metadata.document_count}."
        )

    # --------------------------------------------------------
    # Stage 5:
    # Verify chunk configuration.
    # --------------------------------------------------------

    if metadata.chunk_size != chunk_size:
        raise RuntimeError(
            "Index chunk_size does not match the expected "
            f"configuration: expected {chunk_size}, "
            f"got {metadata.chunk_size}."
        )

    if metadata.chunk_overlap != chunk_overlap:
        raise RuntimeError(
            "Index chunk_overlap does not match the expected "
            f"configuration: expected {chunk_overlap}, "
            f"got {metadata.chunk_overlap}."
        )

    # --------------------------------------------------------
    # Stage 6:
    # Verify retrieval configuration.
    # --------------------------------------------------------

    if metadata.rrf_k != rrf_k:
        raise RuntimeError(
            "Index RRF k does not match the expected "
            f"configuration: expected {rrf_k}, "
            f"got {metadata.rrf_k}."
        )

    # --------------------------------------------------------
    # Stage 7:
    # Verify model configuration.
    # --------------------------------------------------------

    if metadata.embedding_model != settings.embedding_model:
        raise RuntimeError(
            "Index embedding model does not match the current "
            f"configuration: expected {settings.embedding_model}, "
            f"got {metadata.embedding_model}."
        )

    if metadata.reranker_model != settings.reranker_model:
        raise RuntimeError(
            "Index reranker model does not match the current "
            f"configuration: expected {settings.reranker_model}, "
            f"got {metadata.reranker_model}."
        )

    # --------------------------------------------------------
    # Stage 8:
    # Verify the target Qdrant collection.
    # --------------------------------------------------------

    if metadata.qdrant_collection != settings.qdrant_collection:
        raise RuntimeError(
            "Index Qdrant collection does not match the current "
            f"configuration: expected {settings.qdrant_collection}, "
            f"got {metadata.qdrant_collection}."
        )

    # --------------------------------------------------------
    # All checks passed.
    # --------------------------------------------------------

    print("=" * 70)
    print("DocsQuery Index Validation")
    print("=" * 70)
    print("Status:             VALID")
    print(f"Corpus SHA-256:     {manifest.corpus_sha256}")
    print(f"Documents:          {metadata.document_count}")
    print(f"Chunks:             {metadata.chunk_count}")
    print(f"Chunk size:         {metadata.chunk_size}")
    print(f"Chunk overlap:      {metadata.chunk_overlap}")
    print(f"Embedding model:    {metadata.embedding_model}")
    print(f"Reranker model:     {metadata.reranker_model}")
    print(f"RRF k:              {metadata.rrf_k}")
    print(f"Qdrant collection:  {metadata.qdrant_collection}")
    print("=" * 70)


def main() -> None:
    """
    CLI entry point.
    """

    args = parse_args()

    verify_index(
        corpus_root=args.corpus_root,
        manifest_path=args.manifest_path,
        metadata_path=args.metadata_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rrf_k=args.rrf_k,
    )


if __name__ == "__main__":
    main()
