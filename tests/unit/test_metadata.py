"""
Tests for document metadata generation.
"""

from pathlib import Path

import pytest

from app.ingestion.metadata import (
    build_document_metadata,
    generate_document_id,
)


def test_document_id_is_deterministic(tmp_path: Path):
    """
    The same file should always produce the same document ID.
    """

    file_path = tmp_path / "document.txt"

    file_path.write_text("DocsQuery test document.")

    first_id = generate_document_id(str(file_path))

    second_id = generate_document_id(str(file_path))

    assert first_id == second_id


def test_different_content_produces_different_id(
    tmp_path: Path,
):
    """
    Changing the file contents should change its document ID.
    """

    file_path = tmp_path / "document.txt"

    file_path.write_text("Version one.")

    first_id = generate_document_id(str(file_path))

    file_path.write_text("Version two.")

    second_id = generate_document_id(str(file_path))

    assert first_id != second_id


def test_missing_file_raises_error():
    """
    A missing file should raise FileNotFoundError.
    """

    with pytest.raises(FileNotFoundError):
        generate_document_id("does-not-exist.pdf")


def test_build_document_metadata(tmp_path: Path):
    """
    Verify that basic document metadata is created correctly.
    """

    file_path = tmp_path / "manual.pdf"

    file_path.write_bytes(b"test pdf content")

    metadata = build_document_metadata(str(file_path))

    assert len(metadata["document_id"]) == 64
    assert metadata["source"] == "manual.pdf"
    assert metadata["file_type"] == ".pdf"
