"""
DocsQuery - Embedding Service

This module converts text into numerical vectors using a
Sentence Transformers embedding model.

The embedding service is intentionally kept separate from
the ingestion pipeline.

Pipeline:

    DocumentChunk
        ↓
    EmbeddingService
        ↓
    vector
        ↓
    Vector Database
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config.settings import get_settings


class EmbeddingService:
    """
    Generate vector embeddings for text.

    The actual Sentence Transformer model is loaded lazily so
    importing this module does not immediately download/load
    a potentially large model.
    """

    def __init__(self, model_name: str | None = None):
        """
        Initialize the embedding service.

        Args:
            model_name:
                Optional model name.

                If omitted, the model configured in the
                application settings is used.
        """

        settings = get_settings()

        # Use the explicitly supplied model if provided.
        # Otherwise use the configured embedding model.
        self.model_name = model_name or settings.embedding_model

        # The model is loaded lazily by _get_model().
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        """
        Load the embedding model when it is first needed.

        Returns:
            Loaded SentenceTransformer model.
        """

        # Avoid loading the model repeatedly.
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        return self._model

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding for a single piece of text.

        Args:
            text:
                Text to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            ValueError:
                If the text is empty.
        """

        if not text.strip():
            raise ValueError("Cannot generate an embedding for empty text.")

        model = self._get_model()

        # encode() converts text into a numerical vector.
        vector = model.encode(
            text,
            normalize_embeddings=True,
        )

        # Convert NumPy output into normal Python floats.
        return vector.tolist()

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Batch encoding is more efficient than calling
        embed_text() repeatedly.

        Args:
            texts:
                List of text strings.

        Returns:
            List of embedding vectors.
        """

        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Cannot generate embeddings for empty text.")

        model = self._get_model()

        vectors = model.encode(
            texts,
            normalize_embeddings=True,
        )

        return vectors.tolist()

    def dimension(self) -> int:
        """
        Return the dimensionality of the embedding model.

        For example, some models produce vectors with
        384 dimensions, while others may produce 768 or more.

        Returns:
            Number of dimensions in the embedding vector.
        """

        model = self._get_model()

        return model.get_embedding_dimension()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """
    Return a cached EmbeddingService instance.

    This prevents us from creating multiple service objects
    throughout the application.
    """

    return EmbeddingService()
