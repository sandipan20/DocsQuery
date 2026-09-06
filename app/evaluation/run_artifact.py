"""
DocsQuery - Reproducible Evaluation Run Artifact

Stores metadata and outputs belonging to one evaluation execution.

The important design goal is that retrieval and answer evaluation
results can be tied to the same dataset version and Git revision.
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
    """Metadata describing one evaluation execution."""

    run_id: str
    created_at_utc: str

    # Evaluation dataset version used by this run.
    dataset_version: str

    # Source-code revision, when the project is inside a Git repository.
    git_commit: str | None = None

    # Optional fingerprint of a corpus/index file.
    corpus_sha256: str | None = None


class EvaluationRunArtifact(BaseModel):
    """
    Complete machine-readable evaluation run.

    Both retrieval and answer evaluation are stored together so a
    quality gate can reason about one coherent execution.
    """

    metadata: EvaluationRunMetadata

    # Raw output produced by retrieval evaluation.
    retrieval_results: dict[str, Any] = Field(default_factory=dict)

    # Raw output produced by end-to-end RAG evaluation.
    answer_results: dict[str, Any] = Field(default_factory=dict)

    # Small high-level information useful for tooling.
    summary: dict[str, Any] = Field(default_factory=dict)


def make_run_id() -> str:
    """Return a UTC timestamp suitable for identifying a run."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def get_git_commit() -> str | None:
    """
    Return the current Git commit hash.

    Local experiments may run outside Git, so failure is represented
    by None instead of stopping the evaluation.
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
    Return the SHA-256 digest of a file.

    The file is processed incrementally, so large files do not need to
    be loaded completely into memory.
    """

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def save_evaluation_run(
    artifact: EvaluationRunArtifact,
    output_path: str | Path,
) -> None:
    """Save one evaluation run as readable JSON."""

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    """Load a previously saved evaluation run."""

    path = Path(input_path)

    data = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    return EvaluationRunArtifact.model_validate(data)
