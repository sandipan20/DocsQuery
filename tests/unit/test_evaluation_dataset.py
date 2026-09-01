"""
Tests for the evaluation dataset loader.
"""

import json
from pathlib import Path

import pytest

from app.evaluation.dataset import (
    load_evaluation_dataset,
)
from app.retrieval.bm25_storage import BM25Storage


def test_dataset_loads(tmp_path: Path):
    """
    A valid dataset should load successfully.
    """

    dataset_path = tmp_path / "dataset.json"

    dataset_path.write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "examples": [
                    {
                        "example_id": "q001",
                        "query": "What is Python?",
                        "relevant_chunk_ids": ["chunk-001"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = load_evaluation_dataset(str(dataset_path))

    assert dataset.version == "1.0.0"
    assert len(dataset.examples) == 1
    assert dataset.examples[0].query == ("What is Python?")


def test_missing_dataset_fails():
    """
    A missing dataset should raise FileNotFoundError.
    """

    with pytest.raises(FileNotFoundError):
        load_evaluation_dataset("does-not-exist.json")


def test_invalid_dataset_fails(
    tmp_path: Path,
):
    """
    Invalid dataset structure should be rejected.
    """

    dataset_path = tmp_path / "invalid.json"

    dataset_path.write_text(
        json.dumps({"version": "1.0.0"}),
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        load_evaluation_dataset(str(dataset_path))


def test_real_retrieval_dataset_loads():
    """
    Verify that the project's actual evaluation dataset
    is valid.
    """

    dataset = load_evaluation_dataset("data/evaluation/retrieval_dataset.json")

    assert dataset.version == "1.0.0"
    assert len(dataset.examples) >= 3

    for example in dataset.examples:
        assert example.example_id
        assert example.query.strip()
        assert example.relevant_chunk_ids


def test_real_evaluation_chunk_ids_exist():
    """
    Verify that every ground-truth chunk ID in the evaluation
    dataset exists in the indexed corpus.
    """

    # Load the evaluation labels.
    dataset = load_evaluation_dataset("data/evaluation/retrieval_dataset.json")

    # Load the actual indexed chunks.
    storage = BM25Storage("data/index/bm25.json")

    chunks = storage.load()

    # Build a fast lookup set of actual chunk IDs.
    actual_chunk_ids = {chunk.chunk_id for chunk in chunks}

    # Every ground-truth ID must exist in the corpus.
    for example in dataset.examples:
        for chunk_id in example.relevant_chunk_ids:
            assert chunk_id in actual_chunk_ids, (
                f"Evaluation example "
                f"{example.example_id} references "
                f"unknown chunk: {chunk_id}"
            )
