"""
Unit tests for retrieval index validation.
"""

from pathlib import Path

import pytest

from app.storage.corpus_manifest import build_corpus_manifest
from app.storage.index_metadata import create_index_metadata
from scripts.verify_index import verify_index


def create_corpus(root: Path) -> None:
    """
    Create a small test corpus.
    """

    (root / "document_a.pdf").write_bytes(b"PDF A")
    (root / "document_b.pdf").write_bytes(b"PDF B")


def create_metadata(
    corpus_root: Path,
    manifest_path: Path,
    metadata_path: Path,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    rrf_k: int = 60,
    embedding_model: str = "embedding-model",
    reranker_model: str = "reranker-model",
    qdrant_collection: str = "docsquery_chunks",
) -> None:
    """
    Create valid test metadata.
    """

    manifest = build_corpus_manifest(corpus_root)

    metadata = create_index_metadata(
        corpus_sha256=manifest.corpus_sha256,
        corpus_files=manifest.files,
        dataset_version="1.1.0",
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        rrf_k=rrf_k,
        qdrant_collection=qdrant_collection,
        document_count=len(manifest.files),
        chunk_count=20,
    )

    from app.storage.corpus_manifest import save_manifest
    from app.storage.index_metadata import save_index_metadata

    save_manifest(manifest, manifest_path)
    save_index_metadata(metadata, metadata_path)


def patch_settings(monkeypatch):
    """
    Provide deterministic application settings for tests.
    """

    monkeypatch.setattr(
        "scripts.verify_index.get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "embedding_model": "embedding-model",
                "reranker_model": "reranker-model",
                "qdrant_collection": "docsquery_chunks",
            },
        )(),
    )


def test_valid_index_passes(tmp_path, monkeypatch):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()

    create_corpus(corpus_root)

    manifest_path = tmp_path / "manifest.json"
    metadata_path = tmp_path / "metadata.json"

    create_metadata(
        corpus_root,
        manifest_path,
        metadata_path,
    )

    patch_settings(monkeypatch)

    verify_index(
        corpus_root=str(corpus_root),
        manifest_path=str(manifest_path),
        metadata_path=str(metadata_path),
        chunk_size=500,
        chunk_overlap=50,
        rrf_k=60,
    )


def test_changed_corpus_fails_validation(tmp_path, monkeypatch):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()

    create_corpus(corpus_root)

    manifest_path = tmp_path / "manifest.json"
    metadata_path = tmp_path / "metadata.json"

    create_metadata(
        corpus_root,
        manifest_path,
        metadata_path,
    )

    # Modify an existing PDF after the index was supposedly built.
    (corpus_root / "document_a.pdf").write_bytes(b"MODIFIED PDF A")

    patch_settings(monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="does not match the stored corpus manifest",
    ):
        verify_index(
            corpus_root=str(corpus_root),
            manifest_path=str(manifest_path),
            metadata_path=str(metadata_path),
            chunk_size=500,
            chunk_overlap=50,
            rrf_k=60,
        )


def test_wrong_chunk_size_fails(tmp_path, monkeypatch):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()

    create_corpus(corpus_root)

    manifest_path = tmp_path / "manifest.json"
    metadata_path = tmp_path / "metadata.json"

    create_metadata(
        corpus_root,
        manifest_path,
        metadata_path,
        chunk_size=500,
    )

    patch_settings(monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="chunk_size",
    ):
        verify_index(
            corpus_root=str(corpus_root),
            manifest_path=str(manifest_path),
            metadata_path=str(metadata_path),
            chunk_size=400,
            chunk_overlap=50,
            rrf_k=60,
        )


def test_wrong_embedding_model_fails(tmp_path, monkeypatch):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()

    create_corpus(corpus_root)

    manifest_path = tmp_path / "manifest.json"
    metadata_path = tmp_path / "metadata.json"

    create_metadata(
        corpus_root,
        manifest_path,
        metadata_path,
        embedding_model="old-embedding-model",
    )

    patch_settings(monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="embedding model",
    ):
        verify_index(
            corpus_root=str(corpus_root),
            manifest_path=str(manifest_path),
            metadata_path=str(metadata_path),
            chunk_size=500,
            chunk_overlap=50,
            rrf_k=60,
        )


def test_wrong_rrf_value_fails(tmp_path, monkeypatch):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()

    create_corpus(corpus_root)

    manifest_path = tmp_path / "manifest.json"
    metadata_path = tmp_path / "metadata.json"

    create_metadata(
        corpus_root,
        manifest_path,
        metadata_path,
        rrf_k=60,
    )

    patch_settings(monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="RRF k",
    ):
        verify_index(
            corpus_root=str(corpus_root),
            manifest_path=str(manifest_path),
            metadata_path=str(metadata_path),
            chunk_size=500,
            chunk_overlap=50,
            rrf_k=100,
        )
