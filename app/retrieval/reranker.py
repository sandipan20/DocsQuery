"""
DocsQuery - Cross-Encoder Reranker

Takes candidates returned by hybrid retrieval and reranks
them using a cross-encoder model.

Pipeline:

    Query
      +
    Candidate Chunk
      ↓
    Cross Encoder
      ↓
    Relevance Score
"""

from sentence_transformers import CrossEncoder

from app.retrieval.models import RetrievalResult


class CrossEncoderReranker:
    """
    Cross-encoder based document reranker.
    """

    def __init__(
        self,
        model_name: str,
    ):
        """
        Load the cross-encoder model.

        Args:
            model_name:
                Hugging Face cross-encoder model name.
        """

        self.model_name = model_name

        # IMPORTANT:
        # The model is loaded once when the service is created.
        #
        # Do NOT load this inside rerank().
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Rerank retrieval candidates.

        Args:
            query:
                Original user query.

            results:
                Candidate documents from hybrid retrieval.

            top_k:
                Number of results to return.

        Returns:
            Reranked results.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if not results:
            return []

        # Limit top_k to the number of candidates available.
        top_k = min(
            top_k,
            len(results),
        )

        # Build query-document pairs.
        #
        # Cross encoders process the query and document together.
        pairs = [[query, result.text] for result in results]

        # Calculate relevance scores.
        scores = self.model.predict(pairs)

        # Attach reranker scores to results.
        scored_results = [
            result.model_copy(
                update={
                    "score": float(score),
                }
            )
            for result, score in zip(
                results,
                scores,
                strict=True,
            )
        ]

        # Sort by descending relevance.
        scored_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return scored_results[:top_k]
