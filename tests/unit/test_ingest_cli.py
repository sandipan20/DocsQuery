"""
Tests for the document ingestion CLI.
"""

from scripts.ingest_document import parse_arguments


def test_cli_requires_file_path(monkeypatch):
    """
    Verify that the CLI accepts a document path.
    """

    monkeypatch.setattr(
        "sys.argv",
        [
            "ingest_document.py",
            "data/raw/example.pdf",
        ],
    )

    args = parse_arguments()

    assert (
        args.file_path
        == "data/raw/example.pdf"
    )


def test_cli_chunk_configuration(monkeypatch):
    """
    Verify that custom chunk parameters are parsed correctly.
    """

    monkeypatch.setattr(
        "sys.argv",
        [
            "ingest_document.py",
            "data/raw/example.pdf",
            "--chunk-size",
            "400",
            "--overlap",
            "40",
        ],
    )

    args = parse_arguments()

    assert args.chunk_size == 400
    assert args.overlap == 40