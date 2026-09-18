"""
DocsQuery - Retrieval Confidence

Provides semantic confidence checks used to determine whether
a query is sufficiently related to the indexed corpus.
"""

from app.retrieval.models import RetrievalResult


class RetrievalConfidenceGate:
    """
    Determines whether vector retrieval provides enough semantic
    evidence to continue through the RAG pipeline.
    """

    def __init__(
        self,
        vector_threshold: float,
    ) -> None:
        """
        Initialize the confidence gate.

        Args:
            vector_threshold:
                Minimum vector similarity required to continue.
        """

        if not 0.0 <= vector_threshold <= 1.0:
            raise ValueError("vector_threshold must be between 0.0 and 1.0.")

        self.vector_threshold = vector_threshold

    def is_confident(
        self,
        results: list[RetrievalResult],
    ) -> bool:
        """
        Return True when the strongest vector result is strong
        enough to consider the query within the corpus domain.
        """

        if not results:
            return False

        return results[0].score >= self.vector_threshold
