"""
Tests for the DocsQuery embedding service.
"""

import pytest

from app.retrieval.embeddings import EmbeddingService


def test_empty_text_is_rejected():
    """
    Empty text should never be sent to the embedding model.
    """

    service = EmbeddingService()

    with pytest.raises(ValueError):
        service.embed_text("")


def test_whitespace_text_is_rejected():
    """
    Text containing only whitespace should also be rejected.
    """

    service = EmbeddingService()

    with pytest.raises(ValueError):
        service.embed_text("   ")


def test_empty_batch_is_allowed():
    """
    An empty batch should return an empty list.

    This makes batch processing easier for callers.
    """

    service = EmbeddingService()

    assert service.embed_texts([]) == []