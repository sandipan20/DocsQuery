"""
Integration test for real vector retrieval.

Requires:
    - Qdrant running locally
    - embedding model available
    - a previously indexed document
"""

import pytest

from app.retrieval.vector_retriever import VectorRetriever


def test_vector_retrieval_returns_results():
    """
    Verify that a real query can retrieve indexed chunks.
    """

    retriever = VectorRetriever()

    try:
        results = retriever.retrieve(
            "What is Python?",
            limit=3,
        )
    except Exception as exc:
        pytest.fail(f"Vector retrieval failed: {exc}")

    # We expect at least one result because our sample
    # document was indexed earlier.
    assert len(results) > 0

    # Results should be ranked with numerical scores.
    assert all(isinstance(result.score, float) for result in results)

    # Every result must have citation metadata.
    assert all(result.source and result.page_number >= 1 for result in results)
