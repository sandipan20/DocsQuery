"""
Tests for retrieval evaluation dataset quality.

These tests validate the structure and composition of the
retrieval evaluation dataset before retrieval metrics are calculated.
"""

from collections import Counter
from pathlib import Path

import pytest

from app.evaluation.dataset import load_evaluation_dataset
from app.evaluation.models import EvaluationExample

DATASET_PATH = Path("data/evaluation/retrieval_dataset.json")

VALID_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
    "negative",
    "adversarial",
}


def load_dataset():
    """
    Load the real retrieval benchmark.
    """

    return load_evaluation_dataset(DATASET_PATH)


def test_retrieval_dataset_is_not_empty():
    """
    The retrieval evaluation dataset must contain examples.
    """

    dataset = load_dataset()

    assert dataset.examples


def test_non_negative_examples_have_relevant_chunks():
    """
    Positive/adversarial benchmark examples must have at
    least one relevance annotation.
    """

    dataset = load_dataset()

    for example in dataset.examples:
        if example.difficulty != "negative":
            assert example.relevant_chunk_ids


def test_negative_examples_have_no_relevant_chunks():
    """
    Negative examples must explicitly contain zero relevant
    chunks.
    """

    dataset = load_dataset()

    for example in dataset.examples:
        if example.difficulty == "negative":
            assert example.relevant_chunk_ids == []


def test_example_ids_are_unique():
    """
    Example IDs must uniquely identify benchmark queries.
    """

    dataset = load_dataset()

    ids = [example.example_id for example in dataset.examples]

    assert len(ids) == len(set(ids))


def test_queries_are_unique():
    """
    Duplicate benchmark questions should not be present.
    """

    dataset = load_dataset()

    queries = [example.query.strip().lower() for example in dataset.examples]

    assert len(queries) == len(set(queries))


def test_chunk_ids_are_not_empty():
    """
    Every relevance annotation must contain a non-empty
    chunk ID.
    """

    dataset = load_dataset()

    for example in dataset.examples:
        for chunk_id in example.relevant_chunk_ids:
            assert isinstance(chunk_id, str)
            assert chunk_id.strip()


def test_every_example_has_valid_difficulty():
    """
    Every benchmark example must use a supported difficulty.
    """

    dataset = load_dataset()

    for example in dataset.examples:
        assert example.difficulty in VALID_DIFFICULTIES


def test_current_dataset_contains_expected_difficulties():
    """
    The current 48-example benchmark contains:

        easy:         20
        medium:        9
        hard:          3
        negative:     12
        adversarial:   4
    """

    dataset = load_dataset()

    difficulties = Counter(example.difficulty for example in dataset.examples)

    assert len(dataset.examples) == 48

    assert difficulties["easy"] == 20
    assert difficulties["medium"] == 9
    assert difficulties["hard"] == 3
    assert difficulties["negative"] == 12
    assert difficulties["adversarial"] == 4


def test_current_dataset_contains_expected_query_types():
    """
    The benchmark should cover all currently represented
    retrieval topics.
    """

    dataset = load_dataset()

    query_types = {example.query_type for example in dataset.examples}

    assert query_types == {
        "repository-basics",
        "branching",
        "merging",
        "rebasing",
        "remote-repositories",
        "history-inspection",
        "undoing-changes",
        "temporary-work",
        "file-management",
        "tagging",
        "outside-corpus",
    }


def test_current_dataset_contains_expected_query_type_counts():
    """
    Verify the exact composition of the current benchmark
    by query type.
    """

    dataset = load_dataset()

    query_types = Counter(example.query_type for example in dataset.examples)

    assert query_types["repository-basics"] == 7
    assert query_types["remote-repositories"] == 6
    assert query_types["merging"] == 5
    assert query_types["branching"] == 3
    assert query_types["history-inspection"] == 3
    assert query_types["undoing-changes"] == 5
    assert query_types["file-management"] == 3
    assert query_types["temporary-work"] == 2
    assert query_types["rebasing"] == 1
    assert query_types["tagging"] == 1
    assert query_types["outside-corpus"] == 12


def test_expected_total_example_count():
    """
    The retrieval benchmark must contain 40 examples.
    """

    dataset = load_dataset()

    assert len(dataset.examples) == 48


def test_evaluation_examples_are_valid_models():
    """
    Every item in the real dataset should satisfy the
    EvaluationExample Pydantic model.
    """

    dataset = load_dataset()

    for example in dataset.examples:
        assert isinstance(example, EvaluationExample)


def test_negative_example_rules_are_consistent():
    """
    Negative examples must contain no ground-truth relevant chunks.
    """

    dataset = load_dataset()

    for example in dataset.examples:
        if example.difficulty == "negative":
            assert example.relevant_chunk_ids == []


def test_all_referenced_chunk_ids_have_expected_shape():
    """
    Relevant chunk IDs should use the project's chunk-ID format.
    """

    dataset = load_dataset()

    for example in dataset.examples:
        for chunk_id in example.relevant_chunk_ids:
            assert "-chunk-" in chunk_id


@pytest.mark.parametrize(
    "difficulty",
    sorted(VALID_DIFFICULTIES),
)
def test_difficulty_values_are_supported(difficulty: str):
    """
    Each supported difficulty value should be accepted by the
    dataset model.
    """

    relevant_chunks = [] if difficulty == "negative" else ["chunk-test"]

    example = EvaluationExample(
        example_id=f"test-{difficulty}",
        query=f"test query {difficulty}",
        query_type="test",
        difficulty=difficulty,
        relevant_chunk_ids=relevant_chunks,
    )

    assert example.difficulty == difficulty
