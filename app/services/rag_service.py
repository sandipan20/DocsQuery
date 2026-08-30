"""
DocsQuery - RAG Service

Coordinates the complete question-answering pipeline.

Pipeline:

    Query
      ↓
    Retrieval
      ↓
    Reranking
      ↓
    Evidence Check
      ↓
    Generation
      ↓
    Citation Validation
      ↓
    Final Answer
"""

from app.core.timing import measure_time
from app.generation.models import GeneratedAnswer
from app.retrieval.models import RetrievalResult
from app.services.generation_service import (
    GenerationService,
)
from app.services.retrieval_service import (
    RetrievalService,
)


class InsufficientEvidenceError(Exception):
    """
    Raised when the retrieval system cannot provide
    enough evidence to answer safely.
    """


class RAGResponse:
    """
    Internal response from the complete RAG pipeline.
    """

    def __init__(
        self,
        query: str,
        generated_answer: GeneratedAnswer,
        results: list[RetrievalResult],
        retrieval_latency_ms: float,
        generation_latency_ms: float,
        total_latency_ms: float,
    ):
        """
        Initialize a RAG response with observability data.
        """

        self.query = query
        self.generated_answer = generated_answer
        self.results = results

        # Timing information is useful for debugging and
        # production performance monitoring.
        self.retrieval_latency_ms = retrieval_latency_ms

        self.generation_latency_ms = generation_latency_ms

        self.total_latency_ms = total_latency_ms


class RAGService:
    """
    Orchestrates retrieval and answer generation.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        generation_service: GenerationService,
    ):
        """
        Initialize the RAG service.
        """

        self.retrieval_service = retrieval_service

        self.generation_service = generation_service

    def query(
        self,
        query: str,
        top_k: int = 5,
    ) -> RAGResponse:
        """
        Execute the complete RAG pipeline.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        # Start measuring the entire RAG operation.
        with measure_time() as total_timing:
            # ------------------------------------------------
            # Retrieval + reranking
            # ------------------------------------------------

            with measure_time() as retrieval_timing:
                results = self.retrieval_service.search(
                    query=query,
                    limit=top_k,
                )

            # ------------------------------------------------
            # Evidence check
            # ------------------------------------------------

            if not results:
                raise InsufficientEvidenceError("No relevant evidence was found.")

            # ------------------------------------------------
            # Answer generation
            # ------------------------------------------------

            with measure_time() as generation_timing:
                generated_answer = self.generation_service.generate(
                    query=query,
                    results=results,
                )

        return RAGResponse(
            query=query,
            generated_answer=generated_answer,
            results=results,
            retrieval_latency_ms=(retrieval_timing["duration_ms"]),
            generation_latency_ms=(generation_timing["duration_ms"]),
            total_latency_ms=(total_timing["duration_ms"]),
        )
