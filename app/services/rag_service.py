"""
DocsQuery - RAG Service

Coordinates the complete question-answering pipeline.

Pipeline:

    User Query
        ↓
    Retrieval
        ↓
    Reranking
        ↓
    Evidence Check
        ↓
    Context Construction
        ↓
    LLM Generation
        ↓
    Citation Validation
        ↓
    Final Answer
"""

from app.generation.models import GeneratedAnswer
from app.retrieval.models import RetrievalResult
from app.services.generation_service import GenerationService
from app.services.retrieval_service import RetrievalService


class InsufficientEvidenceError(Exception):
    """
    Raised when the retrieval system cannot provide enough
    evidence to answer a question safely.
    """


class RAGResponse:
    """
    Internal representation of a complete RAG response.
    """

    def __init__(
        self,
        query: str,
        generated_answer: GeneratedAnswer,
        results: list[RetrievalResult],
    ):
        """
        Initialize a RAG response.
        """

        self.query = query
        self.generated_answer = generated_answer
        self.results = results


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

        Args:
            query:
                User's question.

            top_k:
                Number of final evidence chunks.

        Returns:
            RAGResponse containing the answer and evidence.

        Raises:
            ValueError:
                For invalid input.

            InsufficientEvidenceError:
                If no evidence is available.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        # ----------------------------------------------------
        # Stage 1:
        # Retrieve and rerank evidence.
        # ----------------------------------------------------

        results = self.retrieval_service.search(
            query=query,
            limit=top_k,
        )

        # ----------------------------------------------------
        # Stage 2:
        # Never ask the LLM to answer without evidence.
        # ----------------------------------------------------

        if not results:
            raise InsufficientEvidenceError("No relevant evidence was found.")
        # ----------------------------------------------------
        # Stage 3:
        # Generate and validate the answer.
        # ----------------------------------------------------

        generated_answer = self.generation_service.generate(
            query=query,
            results=results,
        )

        return RAGResponse(
            query=query,
            generated_answer=generated_answer,
            results=results,
        )
