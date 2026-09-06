"""
Tests for reproducible evaluation run artifacts.
"""

import json

from app.evaluation.run_artifact import (
    EvaluationRunArtifact,
    EvaluationRunMetadata,
    get_git_commit,
    load_evaluation_run,
    make_run_id,
    save_evaluation_run,
)
from app.evaluation.serialization import to_serializable


def test_make_run_id_returns_expected_shape():
    """Run IDs should be UTC timestamp strings."""

    run_id = make_run_id()

    assert len(run_id) == 16
    assert run_id.endswith("Z")
    assert "T" in run_id


def test_metadata_can_be_created():
    """Evaluation metadata should validate correctly."""

    metadata = EvaluationRunMetadata(
        run_id="20260906T120000Z",
        created_at_utc="2026-09-06T12:00:00+00:00",
        dataset_version="1.1.0",
        git_commit="abc123",
    )

    assert metadata.dataset_version == "1.1.0"
    assert metadata.git_commit == "abc123"


def test_artifact_round_trip(tmp_path):
    """Saving and loading an artifact should preserve its data."""

    artifact = EvaluationRunArtifact(
        metadata=EvaluationRunMetadata(
            run_id="20260906T120000Z",
            created_at_utc="2026-09-06T12:00:00+00:00",
            dataset_version="1.1.0",
            git_commit=None,
        ),
        retrieval_results={
            "dataset_version": "1.1.0",
            "metrics": {
                "mrr": 1.0,
            },
        },
        answer_results={
            "dataset_version": "1.1.0",
            "summary": {
                "groundedness": 1.0,
            },
        },
    )

    path = tmp_path / "evaluation_run.json"

    save_evaluation_run(
        artifact,
        path,
    )

    loaded = load_evaluation_run(path)

    assert loaded.metadata.run_id == artifact.metadata.run_id
    assert loaded.metadata.dataset_version == "1.1.0"
    assert loaded.retrieval_results["metrics"]["mrr"] == 1.0


def test_saved_artifact_is_valid_json(tmp_path):
    """The persisted artifact must be ordinary readable JSON."""

    artifact = EvaluationRunArtifact(
        metadata=EvaluationRunMetadata(
            run_id="20260906T120000Z",
            created_at_utc="2026-09-06T12:00:00+00:00",
            dataset_version="1.1.0",
        )
    )

    path = tmp_path / "run.json"

    save_evaluation_run(
        artifact,
        path,
    )

    data = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    assert data["metadata"]["dataset_version"] == "1.1.0"


def test_to_serializable_supports_pydantic_models():
    """Pydantic evaluation objects should become dictionaries."""

    metadata = EvaluationRunMetadata(
        run_id="20260906T120000Z",
        created_at_utc="2026-09-06T12:00:00+00:00",
        dataset_version="1.1.0",
    )

    result = to_serializable(metadata)

    assert isinstance(result, dict)
    assert result["dataset_version"] == "1.1.0"


def test_git_commit_returns_string_or_none():
    """
    Git metadata should never break evaluation just because Git is
    unavailable.
    """

    commit = get_git_commit()

    assert commit is None or isinstance(commit, str)
