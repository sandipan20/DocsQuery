"""
DocsQuery - Index Build Metadata

Stores the exact configuration and corpus information used to
produce the current retrieval indexes.

This allows us to answer:

    "Which corpus and configuration produced this index?"

The metadata is intentionally separate from the actual vector
and BM25 indexes.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.storage.corpus_manifest import CorpusFile

INDEX_METADATA_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class IndexBuildMetadata:
    """
    Metadata describing one complete index build.
    """

    schema_version: str
    corpus_sha256: str
    corpus_files: tuple[CorpusFile, ...]
    dataset_version: str

    embedding_model: str
    reranker_model: str

    chunk_size: int
    chunk_overlap: int

    rrf_k: int

    qdrant_collection: str

    document_count: int
    chunk_count: int

    generated_at: str

    def to_dict(self) -> dict:
        """
        Convert the metadata into a JSON-serializable dictionary.

        The fields here intentionally match the IndexBuildMetadata
        dataclass exactly.

        CorpusFile is a Pydantic model, so model_dump() converts
        each CorpusFile into a normal dictionary.
        """

        return {
            "schema_version": self.schema_version,
            "corpus_sha256": self.corpus_sha256,
            "corpus_files": [
                file.model_dump(mode="json") for file in self.corpus_files
            ],
            "dataset_version": self.dataset_version,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "rrf_k": self.rrf_k,
            "qdrant_collection": self.qdrant_collection,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "generated_at": self.generated_at,
        }


def create_index_metadata(
    *,
    corpus_sha256: str,
    corpus_files: tuple[CorpusFile, ...],
    dataset_version: str,
    embedding_model: str,
    reranker_model: str,
    chunk_size: int,
    chunk_overlap: int,
    rrf_k: int,
    qdrant_collection: str,
    document_count: int,
    chunk_count: int,
) -> IndexBuildMetadata:
    """
    Create metadata for a completed index build.

    The timestamp is generated in UTC so index metadata is
    independent of the local machine timezone.
    """

    if not corpus_sha256.strip():
        raise ValueError("corpus_sha256 cannot be empty.")

    if not corpus_files:
        raise ValueError("corpus_files cannot be empty.")

    if not dataset_version.strip():
        raise ValueError("dataset_version cannot be empty.")

    if not embedding_model.strip():
        raise ValueError("embedding_model cannot be empty.")

    if not reranker_model.strip():
        raise ValueError("reranker_model cannot be empty.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero.")

    if not qdrant_collection.strip():
        raise ValueError("qdrant_collection cannot be empty.")

    if document_count <= 0:
        raise ValueError("document_count must be greater than zero.")

    if chunk_count <= 0:
        raise ValueError("chunk_count must be greater than zero.")

    return IndexBuildMetadata(
        schema_version=INDEX_METADATA_SCHEMA_VERSION,
        corpus_sha256=corpus_sha256,
        corpus_files=corpus_files,
        dataset_version=dataset_version,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        rrf_k=rrf_k,
        qdrant_collection=qdrant_collection,
        document_count=document_count,
        chunk_count=chunk_count,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def save_index_metadata(
    metadata: IndexBuildMetadata,
    output_path: str | Path,
) -> None:
    """
    Persist index metadata as deterministic JSON.
    """

    # Accept both strings and pathlib.Path objects.
    path = Path(output_path)

    # Create the parent directory when necessary.
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # sort_keys=True makes the serialized representation
    # deterministic across runs.
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata.to_dict(),
            file,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")


def load_index_metadata(
    input_path: str | Path,
) -> IndexBuildMetadata:
    """
    Load index metadata from JSON.
    """

    path = Path(input_path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    # Reconstruct each CorpusFile using the actual Pydantic model.
    # CorpusFile currently contains path and sha256.
    corpus_files = tuple(
        CorpusFile.model_validate(file_data) for file_data in data["corpus_files"]
    )

    metadata = IndexBuildMetadata(
        schema_version=data["schema_version"],
        corpus_sha256=data["corpus_sha256"],
        corpus_files=corpus_files,
        dataset_version=data["dataset_version"],
        embedding_model=data["embedding_model"],
        reranker_model=data["reranker_model"],
        chunk_size=data["chunk_size"],
        chunk_overlap=data["chunk_overlap"],
        rrf_k=data["rrf_k"],
        qdrant_collection=data["qdrant_collection"],
        document_count=data["document_count"],
        chunk_count=data["chunk_count"],
        generated_at=data["generated_at"],
    )

    # Validate metadata after loading it from disk.
    validate_index_metadata(metadata)

    return metadata


def validate_index_metadata(
    metadata: IndexBuildMetadata,
) -> None:
    """
    Validate index metadata.

    Raises:
        ValueError:
            If metadata contains an invalid configuration.
    """

    if metadata.schema_version != INDEX_METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported index metadata schema version: {metadata.schema_version}"
        )

    if not metadata.corpus_sha256.strip():
        raise ValueError("corpus_sha256 cannot be empty.")

    if not metadata.corpus_files:
        raise ValueError("corpus_files cannot be empty.")

    if not metadata.dataset_version.strip():
        raise ValueError("dataset_version cannot be empty.")

    if not metadata.embedding_model.strip():
        raise ValueError("embedding_model cannot be empty.")

    if not metadata.reranker_model.strip():
        raise ValueError("reranker_model cannot be empty.")

    if metadata.chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if metadata.chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if metadata.chunk_overlap >= metadata.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if metadata.rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero.")

    if not metadata.qdrant_collection.strip():
        raise ValueError("qdrant_collection cannot be empty.")

    if metadata.document_count <= 0:
        raise ValueError("document_count must be greater than zero.")

    if metadata.chunk_count <= 0:
        raise ValueError("chunk_count must be greater than zero.")

    if not metadata.generated_at.strip():
        raise ValueError("generated_at cannot be empty.")
