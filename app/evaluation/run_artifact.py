"""
Models and helpers for reproducible end-to-end evaluation runs.

An evaluation run records both metadata and evaluation outputs so that
the quality gate can evaluate one coherent snapshot rather than combining
results produced by different commands at different times.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvaluationRunMetadata(BaseModel):
    """Metadata describing exactly what produced an evaluation run."""

    run_id: str
    created_at_utc: str

    dataset_version: str

    git_commit: str | None = None

    corpus_file: str | None = None
    corpus_sha256: str | None = None

    embedding_model: str | None = None
    reranker_model: str | None = None
    groundedness_model: str | None = None
    generation_model: str | None = None

    rrf_k: int | None = None


class EvaluationRunArtifact(BaseModel):
    """
    Complete persisted evaluation result.

    Retrieval and answer evaluation results are stored inside the same
    artifact so the quality gate never mixes results from separate runs.
    """

    metadata: EvaluationRunMetadata

    retrieval_results: dict[str, Any] = Field(default_factory=dict)

    answer_results: dict[str, Any] = Field(default_factory=dict)

    summary: dict[str, Any] = Field(default_factory=dict)


def utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def make_run_id() -> str:
    """Create a sortable UTC-based evaluation run identifier."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def get_git_commit() -> str | None:
    """
    Return the current Git commit hash.

    Evaluation should continue to work even when the project is not
    inside a Git repository, so failures are intentionally converted
    into None.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    commit = result.stdout.strip()

    return commit or None


def sha256_file(path: str | Path) -> str:
    """
    Calculate the SHA-256 hash of a file.

    The hash is calculated in chunks so this also works for large files.
    """

    file_path = Path(path)

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def save_evaluation_run(
    artifact: EvaluationRunArtifact,
    output_path: str | Path,
) -> None:
    """Persist one evaluation run as formatted JSON."""

    path = Path(output_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            artifact.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_evaluation_run(
    input_path: str | Path,
) -> EvaluationRunArtifact:
    """Load a previously persisted evaluation run."""

    path = Path(input_path)

    data = json.loads(path.read_text(encoding="utf-8"))

    return EvaluationRunArtifact.model_validate(data)
