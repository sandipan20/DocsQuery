"""
DocsQuery - BM25 Storage

Provides simple JSON persistence for the BM25 corpus.

We store DocumentChunk metadata and text rather than the
BM25Okapi Python object itself.

On application startup:

    JSON
      ↓
    DocumentChunk objects
      ↓
    BM25Retriever
      ↓
    Searchable BM25 index
"""

import json
from pathlib import Path

from app.ingestion.models import DocumentChunk


class BM25Storage:
    """
    Persist and load BM25 document chunks.

    The storage format is intentionally simple JSON so that
    the data is easy to inspect during development.
    """

    def __init__(self, file_path: str):
        """
        Initialize the storage.

        Args:
            file_path:
                Path where the BM25 corpus will be stored.
        """

        self.file_path = Path(file_path)

    def save(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """
        Save document chunks to disk.

        Args:
            chunks:
                Document chunks that should be persisted.
        """

        # Create the parent directory if it doesn't exist.
        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Convert Pydantic models into JSON-compatible
        # dictionaries.
        data = [chunk.model_dump() for chunk in chunks]

        # Write formatted JSON so it remains human-readable.
        self.file_path.write_text(
            json.dumps(
                data,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self) -> list[DocumentChunk]:
        """
        Load document chunks from disk.

        Returns:
            List of DocumentChunk objects.

        Raises:
            FileNotFoundError:
                If the persistence file doesn't exist.
        """

        if not self.file_path.exists():
            raise FileNotFoundError(f"BM25 storage file not found: {self.file_path}")

        raw_data = json.loads(self.file_path.read_text(encoding="utf-8"))

        return [DocumentChunk.model_validate(item) for item in raw_data]

    def exists(self) -> bool:
        """
        Check whether persisted BM25 data exists.
        """

        return self.file_path.exists()
