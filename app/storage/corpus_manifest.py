"""
Deterministic corpus manifest utilities.

The corpus manifest gives the indexed document collection a stable identity.

Why this exists:
- Retrieval indexes are derived from source documents.
- If PDFs change while an old index is reused, retrieval can become stale.
- A SHA-256 fingerprint lets us detect corpus changes deterministically.

The manifest:
- includes only PDF files
- stores files in deterministic relative-path order
- hashes every PDF
- produces one combined corpus fingerprint
- can be persisted to JSON
- can verify that the original corpus has not changed
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field


class CorpusFile(BaseModel):
    """Metadata for one file in the corpus."""

    path: str
    sha256: str


class CorpusManifest(BaseModel):
    """Deterministic description of a corpus."""

    corpus_sha256: str
    files: list[CorpusFile] = Field(default_factory=list)
    corpus_dir: str


def _sha256_file(path: Path) -> str:
    """
    Return the SHA-256 hash of a file.

    The file is read in chunks so large PDFs do not need to be loaded
    entirely into memory.
    """
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def build_corpus_manifest(
    corpus_dir: Path,
) -> CorpusManifest:
    """
    Build a deterministic manifest for all PDFs under corpus_dir.

    Files are sorted by their relative path, making the generated
    corpus fingerprint independent of filesystem traversal order.
    """
    corpus_dir = corpus_dir.resolve()

    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    pdf_files = sorted(
        (
            path
            for path in corpus_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.relative_to(corpus_dir).as_posix(),
    )

    if not pdf_files:
        raise ValueError(f"Corpus directory contains no PDF files: {corpus_dir}")

    files = [
        CorpusFile(
            path=path.relative_to(corpus_dir).as_posix(),
            sha256=_sha256_file(path),
        )
        for path in pdf_files
    ]

    # Build the combined corpus fingerprint from stable
    # path/hash pairs.
    digest = hashlib.sha256()

    for file_entry in files:
        digest.update(file_entry.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_entry.sha256.encode("ascii"))
        digest.update(b"\n")

    return CorpusManifest(
        corpus_sha256=digest.hexdigest(),
        files=files,
        corpus_dir=str(corpus_dir),
    )


def save_manifest(
    manifest: CorpusManifest,
    output_path: str | Path,
) -> None:
    """
    Save a corpus manifest as deterministic JSON.

    Args:
        manifest:
            The Pydantic corpus manifest to save.
        output_path:
            Destination path. Both str and pathlib.Path are accepted.
    """
    # Normalize the path so we can safely use .parent and .write_text().
    output_path = Path(output_path)

    # Make sure the destination directory exists.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # CorpusManifest is a Pydantic model, so use model_dump()
    # instead of assuming it has a custom to_dict() method.
    manifest_data = manifest.model_dump(
        mode="json",
    )

    # sort_keys + stable indentation makes the output deterministic.
    output_path.write_text(
        json.dumps(
            manifest_data,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_manifest(
    manifest_path: str | Path,
) -> CorpusManifest:
    """
    Load and validate a previously saved manifest.

    Both string paths and pathlib.Path objects are accepted.
    """
    # Normalize the incoming value so callers can safely pass
    # either a string or a pathlib.Path.
    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    return CorpusManifest.model_validate(data)


def verify_manifest(
    manifest: CorpusManifest,
) -> bool:
    """
    Verify that the corpus represented by a manifest
    has not changed.

    Returns:
        True when the current corpus matches the manifest.

    Raises:
        ValueError: When the corpus has changed or the corpus
        can no longer be verified.
    """
    corpus_dir = Path(manifest.corpus_dir)

    try:
        current = build_corpus_manifest(corpus_dir)
    except (
        FileNotFoundError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Corpus verification failed: the corpus is missing "
            "or contains no PDF files."
        ) from exc

    if current.corpus_sha256 != manifest.corpus_sha256:
        raise ValueError(
            "Corpus verification failed: the corpus has changed "
            "since the manifest was created."
        )

    return True
