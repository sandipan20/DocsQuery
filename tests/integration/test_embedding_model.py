"""
Integration test for the real embedding model.

Unlike the unit tests, this test actually loads the configured
Sentence Transformer model and generates an embedding.
"""

from app.retrieval.embeddings import EmbeddingService


def test_real_embedding_generation():
    """
    Verify that the configured embedding model can generate
    a valid vector.
    """

    service = EmbeddingService()

    vector = service.embed_text(
        "DocsQuery is a document question answering system."
    )

    # The vector should contain numerical values.
    assert len(vector) > 0

    assert all(
        isinstance(value, float)
        for value in vector
    )

    # Verify that the returned dimension matches the model.
    assert len(vector) == service.dimension()