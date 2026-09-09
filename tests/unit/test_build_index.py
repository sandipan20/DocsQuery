"""
Tests for the canonical DocsQuery index builder.
"""

from pathlib import Path

import pytest

from app.ingestion.models import DocumentChunk
from scripts.build_index import build_index


class FakeIndexManager:
    """
    Fake index manager used to avoid loading real ML models
    during unit tests.
    """

    def __init__(self):
        self.received_chunks = []

    def rebuild(
        self,
        chunks: list[DocumentChunk],
    ) -> int:
        """
        Record the chunks received by the index builder.
        """

        self.received_chunks = chunks

        return len(chunks)


def test_build_index_builds_from_all_pdf_files(
    tmp_path,
    monkeypatch,
):
    """
    Every PDF discovered in the corpus should be ingested and
    passed to the index manager.
    """

    corpus = tmp_path / "corpus"

    corpus.mkdir()

    (corpus / "a.pdf").write_bytes(b"fake pdf a")

    (corpus / "b.pdf").write_bytes(b"fake pdf b")

    fake_manager = FakeIndexManager()

    def fake_ingest_pdf(
        file_path: str,
        chunk_size: int,
        overlap: int,
    ):
        """
        Return one deterministic test chunk per PDF.
        """

        path = Path(file_path)

        return [
            DocumentChunk(
                chunk_id=f"{path.stem}-chunk-0",
                document_id=path.stem,
                text="test content",
                source=path.name,
                page_number=1,
                chunk_index=0,
            )
        ]

    # Replace real PDF ingestion so this test remains fast.
    monkeypatch.setattr(
        "scripts.build_index.ingest_pdf",
        fake_ingest_pdf,
    )

    # Replace real BM25/Qdrant management.
    monkeypatch.setattr(
        "scripts.build_index.RetrievalIndexManager",
        lambda: fake_manager,
    )

    # Use lightweight fake settings.
    monkeypatch.setattr(
        "scripts.build_index.get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "embedding_model": ("test-embedding-model"),
                "reranker_model": ("test-reranker-model"),
                "qdrant_collection": ("test-collection"),
            },
        )(),
    )

    manifest_path = tmp_path / "manifest.json"

    metadata_path = tmp_path / "metadata.json"

    result = build_index(
        corpus_root=str(corpus),
        manifest_path=str(manifest_path),
        metadata_path=str(metadata_path),
        chunk_size=500,
        chunk_overlap=50,
        dataset_version="1.1.0",
        rrf_k=60,
    )

    assert result == 2

    assert len(fake_manager.received_chunks) == 2

    assert manifest_path.exists()
    assert metadata_path.exists()


def test_invalid_chunk_configuration_is_rejected(
    tmp_path,
):
    """
    Invalid chunk configuration must fail before indexing.
    """

    corpus = tmp_path / "corpus"

    corpus.mkdir()

    (corpus / "document.pdf").write_bytes(b"fake")

    with pytest.raises(
        ValueError,
        match="chunk_overlap",
    ):
        build_index(
            corpus_root=str(corpus),
            manifest_path=str(tmp_path / "manifest.json"),
            metadata_path=str(tmp_path / "metadata.json"),
            chunk_size=100,
            chunk_overlap=100,
            dataset_version="1.1.0",
            rrf_k=60,
        )


def test_empty_ingestion_is_rejected(
    tmp_path,
    monkeypatch,
):
    """
    A corpus producing zero chunks must never produce an
    apparently successful empty index.
    """

    corpus = tmp_path / "corpus"

    corpus.mkdir()

    (corpus / "document.pdf").write_bytes(b"fake")

    monkeypatch.setattr(
        "scripts.build_index.ingest_pdf",
        lambda **kwargs: [],
    )

    with pytest.raises(
        RuntimeError,
        match="zero chunks",
    ):
        build_index(
            corpus_root=str(corpus),
            manifest_path=str(tmp_path / "manifest.json"),
            metadata_path=str(tmp_path / "metadata.json"),
            chunk_size=500,
            chunk_overlap=50,
            dataset_version="1.1.0",
            rrf_k=60,
        )
