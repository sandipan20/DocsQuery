"""
Unit tests for deterministic corpus manifest generation.
"""

from pathlib import Path

import pytest

from app.storage.corpus_manifest import (
    build_corpus_manifest,
    load_manifest,
    save_manifest,
    verify_manifest,
)


def test_manifest_is_deterministic(
    tmp_path: Path,
):
    """
    The same corpus must produce the same fingerprint.
    """

    (tmp_path / "b.pdf").write_bytes(b"document b")

    (tmp_path / "a.pdf").write_bytes(b"document a")

    first = build_corpus_manifest(tmp_path)

    second = build_corpus_manifest(tmp_path)

    assert first.corpus_sha256 == second.corpus_sha256

    assert [file.path for file in first.files] == [
        "a.pdf",
        "b.pdf",
    ]


def test_manifest_changes_when_file_changes(
    tmp_path: Path,
):
    """
    Modifying a corpus file must change the corpus fingerprint.
    """

    pdf = tmp_path / "document.pdf"

    pdf.write_bytes(b"version one")

    first = build_corpus_manifest(tmp_path)

    pdf.write_bytes(b"version two")

    second = build_corpus_manifest(tmp_path)

    assert first.corpus_sha256 != second.corpus_sha256


def test_manifest_changes_when_file_is_added(
    tmp_path: Path,
):
    """
    Adding another corpus document must change the fingerprint.
    """

    (tmp_path / "a.pdf").write_bytes(b"document a")

    first = build_corpus_manifest(tmp_path)

    (tmp_path / "b.pdf").write_bytes(b"document b")

    second = build_corpus_manifest(tmp_path)

    assert first.corpus_sha256 != second.corpus_sha256


def test_non_pdf_files_are_ignored(
    tmp_path: Path,
):
    """
    The default corpus manifest should only include PDFs.
    """

    (tmp_path / "document.pdf").write_bytes(b"pdf content")

    (tmp_path / "notes.txt").write_text(
        "not part of the PDF corpus",
        encoding="utf-8",
    )

    manifest = build_corpus_manifest(tmp_path)

    assert len(manifest.files) == 1

    assert manifest.files[0].path == "document.pdf"


def test_empty_corpus_is_rejected(
    tmp_path: Path,
):
    """
    An empty corpus must never silently produce an empty index.
    """

    with pytest.raises(ValueError):
        build_corpus_manifest(tmp_path)


def test_manifest_round_trip(
    tmp_path: Path,
):
    """
    Saved and loaded manifests must preserve their contents.
    """

    (tmp_path / "document.pdf").write_bytes(b"document")

    manifest = build_corpus_manifest(tmp_path)

    output = tmp_path / "manifest.json"

    save_manifest(
        manifest,
        output,
    )

    loaded = load_manifest(output)

    assert loaded.corpus_sha256 == manifest.corpus_sha256

    assert loaded.files == manifest.files


def test_manifest_verification_passes(
    tmp_path: Path,
):
    """
    An unchanged corpus must successfully verify.
    """

    (tmp_path / "document.pdf").write_bytes(b"document")

    manifest = build_corpus_manifest(tmp_path)

    assert verify_manifest(manifest) is True


def test_manifest_verification_fails_after_change(
    tmp_path: Path,
):
    """
    Verification must detect corpus mutations.
    """

    pdf = tmp_path / "document.pdf"

    pdf.write_bytes(b"original")

    manifest = build_corpus_manifest(tmp_path)

    pdf.write_bytes(b"changed")

    with pytest.raises(ValueError):
        verify_manifest(manifest)


def test_load_manifest_accepts_string_path(tmp_path):
    """load_manifest should accept both str and Path inputs."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    pdf_path = corpus_dir / "example.pdf"
    pdf_path.write_bytes(b"test pdf content")

    manifest = build_corpus_manifest(corpus_dir)

    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)

    loaded = load_manifest(str(manifest_path))

    assert loaded == manifest
