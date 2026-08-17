"""
DocsQuery - Document Metadata

This module creates stable identifiers and metadata for source
documents.

A document ID is generated from the file contents using SHA-256.

Same file content
    -> same document ID

Changed file content
    -> different document ID
"""

import hashlib
from pathlib import Path


def generate_document_id(file_path: str) -> str:
    """
    Generate a deterministic ID from the document contents.

    Args:
        file_path:
            Path to the source document.

    Returns:
        A SHA-256 hexadecimal identifier.

    Raises:
        FileNotFoundError:
            If the document does not exist.
    """

    path = Path(file_path)

    # Fail early if the document doesn't exist.
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    # SHA-256 is used as a content fingerprint.
    hasher = hashlib.sha256()

    # Read the file in chunks instead of loading the entire
    # document into memory at once.
    with path.open("rb") as file:
        while data := file.read(1024 * 1024):
            hasher.update(data)

    return hasher.hexdigest()


def build_document_metadata(file_path: str) -> dict[str, str]:
    """
    Build basic metadata for a source document.

    Args:
        file_path:
            Path to the source document.

    Returns:
        Dictionary containing stable document information.
    """

    path = Path(file_path)

    return {
        "document_id": generate_document_id(file_path),
        "source": path.name,
        "file_type": path.suffix.lower(),
    }
