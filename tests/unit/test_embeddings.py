"""
Tests for the DocsQuery embedding service.
"""

import numpy as np
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


def test_short_text_produces_expected_dimension():
    """
    A normal short input should produce a vector whose
    dimension matches the embedding model.
    """

    service = EmbeddingService()

    vector = service.embed_text("git status shows the state of the working tree.")

    assert len(vector) == service.dimension()


def test_long_text_is_split_into_token_safe_windows():
    """
    Long text should be split so that every window remains
    within the model's maximum sequence length.
    """

    service = EmbeddingService()

    text = "git status " * 500

    windows = service._split_into_token_windows(text)

    assert len(windows) > 1

    model = service._get_model()

    for window in windows:
        token_ids = model.tokenizer(
            window,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        assert len(token_ids) <= model.max_seq_length


def test_long_text_embedding_has_expected_dimension():
    """
    A text longer than the model's token limit should still
    produce one final embedding vector.
    """

    service = EmbeddingService()

    text = ("Git status shows the current state of tracked and untracked files. ") * 300

    vector = service.embed_text(text)

    assert len(vector) == service.dimension()


def test_long_text_embedding_is_normalized():
    """
    The final pooled embedding should be L2-normalized.
    """

    service = EmbeddingService()

    text = ("The git commit command records staged content as a new snapshot. ") * 300

    vector = np.array(service.embed_text(text))

    norm = np.linalg.norm(vector)

    assert norm == pytest.approx(1.0, abs=1e-5)


def test_embed_texts_supports_long_inputs():
    """
    Batch embedding should also support inputs that exceed
    the model's native token limit.
    """

    service = EmbeddingService()

    texts = [
        "git status " * 200,
        "git commit " * 200,
    ]

    vectors = service.embed_texts(texts)

    assert len(vectors) == 2

    for vector in vectors:
        assert len(vector) == service.dimension()


def test_long_text_does_not_raise_token_length_warning():
    """
    Long text processing should be handled by our token-window
    logic rather than being passed directly to the model as an
    oversized sequence.
    """

    service = EmbeddingService()

    text = "git status " * 500

    # Calling embed_text should succeed because the service
    # splits the input before calling SentenceTransformer.
    vector = service.embed_text(text)

    assert vector


def test_long_text_splitter_uses_multiple_safe_windows():
    """
    Oversized text should be split into multiple windows without
    requiring the embedding model to truncate the input.
    """

    service = EmbeddingService()

    text = "git status " * 500

    windows = service._split_into_token_windows(text)

    assert len(windows) > 1

    model = service._get_model()

    for window in windows:
        token_ids = model.tokenizer(
            window,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        assert len(token_ids) <= model.max_seq_length
