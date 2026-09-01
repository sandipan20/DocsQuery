"""
Tests for corpus ingestion utilities.
"""

from pathlib import Path

from scripts.ingest_corpus import find_pdfs


def test_find_pdfs_returns_only_pdf_files(
    tmp_path: Path,
):
    """
    Only PDF files should be included.
    """

    (tmp_path / "a.pdf").touch()
    (tmp_path / "b.PDF").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "data.json").touch()

    pdfs = find_pdfs(tmp_path)

    assert [path.name for path in pdfs] == [
        "a.pdf",
        "b.PDF",
    ]


def test_find_pdfs_returns_sorted_files(
    tmp_path: Path,
):
    """
    PDF discovery should be deterministic.
    """

    (tmp_path / "c.pdf").touch()
    (tmp_path / "a.pdf").touch()
    (tmp_path / "b.pdf").touch()

    pdfs = find_pdfs(tmp_path)

    assert [path.name for path in pdfs] == [
        "a.pdf",
        "b.pdf",
        "c.pdf",
    ]


def test_empty_directory_has_no_pdfs(
    tmp_path: Path,
):
    """
    An empty directory should return an empty list.
    """

    assert find_pdfs(tmp_path) == []
