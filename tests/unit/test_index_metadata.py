"""
Unit tests for index build metadata.
"""

from pathlib import Path

import pytest

from app.storage.corpus_manifest import CorpusFile
from app.storage.index_metadata import (
    INDEX_METADATA_SCHEMA_VERSION,
    IndexBuildMetadata,
    create_index_metadata,
    load_index_metadata,
    save_index_metadata,
    validate_index_metadata,
)


def sample_metadata() -> IndexBuildMetadata:
    """
    Return valid deterministic metadata for testing.
    """

    return IndexBuildMetadata(
        schema_version=INDEX_METADATA_SCHEMA_VERSION,
        corpus_sha256="abc123",
        corpus_files=(
            CorpusFile(
                path="document.pdf",
                sha256="filehash",
                size_bytes=100,
            ),
        ),
        dataset_version="1.1.0",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunk_size=500,
        chunk_overlap=50,
        rrf_k=60,
        qdrant_collection="docsquery_chunks",
        document_count=1,
        chunk_count=10,
        generated_at="2026-01-01T00:00:00+00:00",
    )


def test_metadata_round_trip(tmp_path: Path):
    metadata = sample_metadata()

    path = tmp_path / "index_metadata.json"

    save_index_metadata(metadata, path)

    loaded = load_index_metadata(path)

    assert loaded == metadata


def test_metadata_json_is_deterministic(tmp_path: Path):
    metadata = sample_metadata()

    path1 = tmp_path / "metadata1.json"
    path2 = tmp_path / "metadata2.json"

    save_index_metadata(metadata, path1)
    save_index_metadata(metadata, path2)

    assert path1.read_text(encoding="utf-8") == path2.read_text(encoding="utf-8")


def test_valid_metadata_passes_validation():
    metadata = sample_metadata()

    validate_index_metadata(metadata)


def test_invalid_chunk_configuration():
    with pytest.raises(ValueError, match="chunk_overlap"):
        create_index_metadata(
            corpus_sha256="abc123",
            corpus_files=(
                CorpusFile(
                    path="document.pdf",
                    sha256="hash",
                    size_bytes=100,
                ),
            ),
            dataset_version="1.1.0",
            embedding_model="embedding-model",
            reranker_model="reranker-model",
            chunk_size=100,
            chunk_overlap=100,
            rrf_k=60,
            qdrant_collection="docsquery_chunks",
            document_count=1,
            chunk_count=10,
        )


def test_empty_corpus_hash_is_rejected():
    with pytest.raises(ValueError, match="corpus_sha256"):
        create_index_metadata(
            corpus_sha256="",
            corpus_files=(
                CorpusFile(
                    path="document.pdf",
                    sha256="hash",
                    size_bytes=100,
                ),
            ),
            dataset_version="1.1.0",
            embedding_model="embedding-model",
            reranker_model="reranker-model",
            chunk_size=500,
            chunk_overlap=50,
            rrf_k=60,
            qdrant_collection="docsquery_chunks",
            document_count=1,
            chunk_count=10,
        )


def test_empty_index_is_rejected():
    with pytest.raises(ValueError, match="chunk_count"):
        create_index_metadata(
            corpus_sha256="abc123",
            corpus_files=(
                CorpusFile(
                    path="document.pdf",
                    sha256="hash",
                    size_bytes=100,
                ),
            ),
            dataset_version="1.1.0",
            embedding_model="embedding-model",
            reranker_model="reranker-model",
            chunk_size=500,
            chunk_overlap=50,
            rrf_k=60,
            qdrant_collection="docsquery_chunks",
            document_count=1,
            chunk_count=0,
        )
