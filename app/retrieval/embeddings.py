"""
DocsQuery - Embedding Service

This module converts text into numerical vectors using a
Sentence Transformers embedding model.

Long document chunks are split into token-safe windows before
embedding so that text beyond the model's maximum sequence
length is not silently discarded.

Pipeline:

    DocumentChunk
        ↓
    token-safe windows
        ↓
    EmbeddingService
        ↓
    vector
        ↓
    Vector Database
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config.settings import get_settings


class EmbeddingService:
    """
    Generate vector embeddings for text.

    Long inputs are split into token-safe windows before
    embedding. Window embeddings are mean-pooled and the
    final vector is normalized.

    This preserves the existing DocumentChunk boundaries and
    chunk IDs while preventing embedding truncation.
    """

    def __init__(self, model_name: str | None = None):
        """
        Initialize the embedding service.

        Args:
            model_name:
                Optional model name.

                If omitted, the model configured in application
                settings is used.
        """

        settings = get_settings()

        self.model_name = model_name or settings.embedding_model

        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        """
        Load the embedding model lazily.

        Returns:
            Loaded SentenceTransformer model.
        """

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        return self._model

    def _split_into_token_windows(
        self,
        text: str,
    ) -> list[str]:
        """
        Split text into windows that fit the model's token limit.

        The underlying fast tokenizer is temporarily configured with
        truncation disabled so that the complete token sequence can be
        inspected before we perform the manual split.

        Args:
            text:
                Text to split.

        Returns:
            Token-safe text windows.
        """

        model = self._get_model()
        tokenizer = model.tokenizer
        backend_tokenizer = tokenizer.backend_tokenizer

        # Save the current truncation configuration.
        previous_truncation = backend_tokenizer.truncation

        # Disable backend truncation so we can inspect the full sequence.
        backend_tokenizer.no_truncation()

        try:
            encoding = backend_tokenizer.encode(
                text,
                add_special_tokens=False,
            )

            token_ids = encoding.ids

        finally:
            # Restore the tokenizer's previous truncation configuration.
            if previous_truncation is None:
                backend_tokenizer.no_truncation()
            else:
                backend_tokenizer.enable_truncation(**previous_truncation)

        if not token_ids:
            return []

        # Leave room for special tokens added by the model.
        max_length = max(model.max_seq_length - 4, 1)

        windows: list[str] = []

        for start in range(0, len(token_ids), max_length):
            window_ids = token_ids[start : start + max_length]

            window_text = tokenizer.decode(
                window_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            if window_text:
                windows.append(window_text)

        return windows

    def _embed_long_text(
        self,
        text: str,
    ) -> list[float]:
        """
        Embed text using token-safe windows.

        Each window is embedded independently. The resulting
        vectors are mean-pooled and normalized.

        Args:
            text:
                Text to embed.

        Returns:
            Normalized embedding vector.
        """

        model = self._get_model()

        windows = self._split_into_token_windows(text)

        if not windows:
            raise ValueError("Cannot generate an embedding for empty text.")

        # A single safe window can use the normal encode path.
        if len(windows) == 1:
            vector = model.encode(
                windows[0],
                normalize_embeddings=True,
            )

            return vector.tolist()

        # Embed every window independently.
        window_vectors = model.encode(
            windows,
            normalize_embeddings=False,
        )

        # Mean-pool all window representations.
        pooled_vector = np.mean(
            window_vectors,
            axis=0,
        )

        # Normalize the pooled representation.
        norm = np.linalg.norm(pooled_vector)

        if norm == 0:
            raise ValueError("Generated embedding has zero magnitude.")

        pooled_vector = pooled_vector / norm

        return pooled_vector.tolist()

    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a single piece of text.

        Long inputs are automatically split into token-safe
        windows.

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

        return self._embed_long_text(text)

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Each text is independently split into token-safe windows.

        Args:
            texts:
                List of text strings.

        Returns:
            List of embedding vectors.

        Raises:
            ValueError:
                If any text is empty.
        """

        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Cannot generate embeddings for empty text.")

        return [self._embed_long_text(text) for text in texts]

    def dimension(self) -> int:
        """
        Return the dimensionality of the embedding model.

        Returns:
            Number of dimensions in the embedding vector.
        """

        model = self._get_model()

        return model.get_embedding_dimension()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """
    Return a cached EmbeddingService instance.

    This prevents multiple embedding service objects from
    being created throughout the application.
    """

    return EmbeddingService()
