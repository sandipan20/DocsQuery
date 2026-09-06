"""
DocsQuery - Unified Reproducible Evaluation Runner

Runs the existing retrieval and end-to-end RAG evaluators in one
logical execution and combines their outputs into:

    data/evaluation/evaluation_run.json

The existing evaluators remain responsible for the actual evaluation
logic. This script is only the orchestration layer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from app.evaluation.dataset import load_evaluation_dataset
from app.evaluation.run_artifact import (
    EvaluationRunArtifact,
    EvaluationRunMetadata,
    get_git_commit,
    make_run_id,
    save_evaluation_run,
    sha256_file,
)
from app.evaluation.serialization import to_serializable

DEFAULT_DATASET = Path("data/evaluation/retrieval_dataset.json")

DEFAULT_OUTPUT = Path("data/evaluation/evaluation_run.json")

RETRIEVAL_OUTPUT = Path("data/evaluation/retrieval_results.json")

RAG_OUTPUT = Path("data/evaluation/rag_results.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run retrieval and end-to-end RAG evaluation as one reproducible run."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the evaluation dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the combined evaluation artifact.",
    )

    parser.add_argument(
        "--corpus-file",
        type=Path,
        default=None,
        help=(
            "Optional file to fingerprint with SHA-256. "
            "Useful for recording the evaluated corpus/index."
        ),
    )

    return parser.parse_args()


def remove_previous_outputs() -> None:
    """
    Remove stale evaluator outputs before starting.

    This is critical: we must never accidentally combine a newly
    generated retrieval result with an old RAG result.
    """

    for path in (
        RETRIEVAL_OUTPUT,
        RAG_OUTPUT,
    ):
        path.unlink(
            missing_ok=True,
        )


def run_module(module_name: str) -> None:
    """
    Run an existing package-style script.

    Using `python -m ...` preserves the project's package execution
    convention.
    """

    subprocess.run(
        [
            sys.executable,
            "-m",
            module_name,
        ],
        check=True,
    )


def load_json(path: Path) -> dict:
    """Load a JSON object from disk."""

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def require_output(path: Path) -> dict:
    """Fail clearly if an expected evaluator output was not produced."""

    if not path.exists():
        raise RuntimeError(f"Expected evaluation output was not created: {path}")

    return load_json(path)


def extract_dataset_version(
    data: dict,
    *,
    source_name: str,
) -> str:
    """
    Extract the dataset version from an evaluator result.

    Different evaluator result formats may put this in slightly
    different locations, so the helper keeps that compatibility logic
    in one place.
    """

    # Most current result formats expose dataset_version directly.
    version = data.get("dataset_version")

    if version:
        return str(version)

    # Some structured results store the summary under `summary`.
    summary = data.get("summary")

    if isinstance(summary, dict):
        version = summary.get("dataset_version")

        if version:
            return str(version)

    raise RuntimeError(f"Could not find dataset_version in {source_name}.")


def main() -> None:
    """Run the complete reproducible evaluation."""

    args = parse_args()

    # ------------------------------------------------------------
    # Validate the dataset before doing expensive work.
    # ------------------------------------------------------------

    dataset = load_evaluation_dataset(str(args.dataset))

    print()
    print("=" * 70)
    print("DocsQuery Reproducible Evaluation")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Dataset version: {dataset.version}")
    print(f"Examples: {len(dataset.examples)}")
    print()

    # ------------------------------------------------------------
    # Remove results from previous executions.
    #
    # This prevents stale-result mixing.
    # ------------------------------------------------------------

    remove_previous_outputs()

    # ------------------------------------------------------------
    # Run deterministic retrieval evaluation.
    # ------------------------------------------------------------

    print("Running retrieval evaluation...")
    run_module("scripts.evaluate_retrieval")

    retrieval_results = require_output(RETRIEVAL_OUTPUT)

    retrieval_dataset_version = extract_dataset_version(
        retrieval_results,
        source_name=str(RETRIEVAL_OUTPUT),
    )

    # ------------------------------------------------------------
    # Run end-to-end RAG evaluation.
    #
    # This requires the Gemini API and therefore is not a
    # lightweight unit-test operation.
    # ------------------------------------------------------------

    print()
    print("Running end-to-end RAG evaluation...")
    run_module("scripts.evaluate_rag")

    answer_results = require_output(RAG_OUTPUT)

    answer_dataset_version = extract_dataset_version(
        answer_results,
        source_name=str(RAG_OUTPUT),
    )

    # ------------------------------------------------------------
    # Dataset-version consistency check.
    # ------------------------------------------------------------

    expected_version = str(dataset.version)

    if retrieval_dataset_version != expected_version:
        raise RuntimeError(
            "Retrieval evaluation used a different dataset version: "
            f"{retrieval_dataset_version} != {expected_version}"
        )

    if answer_dataset_version != expected_version:
        raise RuntimeError(
            "RAG evaluation used a different dataset version: "
            f"{answer_dataset_version} != {expected_version}"
        )

    if retrieval_dataset_version != answer_dataset_version:
        raise RuntimeError(
            "Retrieval and RAG evaluations used different dataset versions."
        )

    # ------------------------------------------------------------
    # Optional corpus fingerprint.
    # ------------------------------------------------------------

    corpus_hash = None

    if args.corpus_file is not None:
        corpus_hash = sha256_file(args.corpus_file)

    # ------------------------------------------------------------
    # Construct the single run artifact.
    # ------------------------------------------------------------

    metadata = EvaluationRunMetadata(
        run_id=make_run_id(),
        created_at_utc=(
            __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat()
        ),
        dataset_version=expected_version,
        git_commit=get_git_commit(),
        corpus_sha256=corpus_hash,
    )

    artifact = EvaluationRunArtifact(
        metadata=metadata,
        retrieval_results=to_serializable(retrieval_results),
        answer_results=to_serializable(answer_results),
        summary={
            "evaluation_completed": True,
            "dataset_version": expected_version,
            "num_examples": len(dataset.examples),
        },
    )

    save_evaluation_run(
        artifact,
        args.output,
    )

    print()
    print("=" * 70)
    print("Evaluation run completed successfully.")
    print("=" * 70)
    print(f"Run ID:        {metadata.run_id}")
    print(f"Dataset:       {metadata.dataset_version}")
    print(f"Git commit:    {metadata.git_commit}")
    print(f"Output:        {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
