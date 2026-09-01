"""
Tests for retrieval diagnostics.
"""

from app.evaluation.diagnostics import (
    analyze_example,
)


def test_analyze_example_detects_hits():
    """
    A strategy should be marked as a hit when it retrieves
    at least one relevant chunk.
    """

    diagnostic = analyze_example(
        example_id="q001",
        query="What is Git?",
        relevant_chunk_ids=[
            "chunk-1",
        ],
        strategy_results={
            "BM25": [
                "chunk-1",
                "chunk-2",
            ],
            "Vector": [
                "chunk-3",
                "chunk-4",
            ],
        },
    )

    bm25 = diagnostic.strategies[0]
    vector = diagnostic.strategies[1]

    assert bm25.hit is True
    assert vector.hit is False


def test_diagnostic_preserves_ranked_results():
    """
    Diagnostics should preserve the original ranking order.
    """

    diagnostic = analyze_example(
        example_id="q001",
        query="Git branch",
        relevant_chunk_ids=[
            "chunk-2",
        ],
        strategy_results={
            "BM25": [
                "chunk-3",
                "chunk-2",
                "chunk-1",
            ],
        },
    )

    assert diagnostic.strategies[0].retrieved_chunk_ids == [
        "chunk-3",
        "chunk-2",
        "chunk-1",
    ]
