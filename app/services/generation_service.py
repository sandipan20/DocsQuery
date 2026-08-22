"""
DocsQuery - Generation Service

Coordinates:

    Retrieved Evidence
          ↓
    Context Builder
          ↓
        Gemini
          ↓
    Citation Validator
          ↓
    Generated Answer
"""

from app.generation.citation_validator import (
    CitationValidator,
)
from app.generation.context_builder import (
    ContextBuilder,
)
from app.generation.llm import LLMService
from app.generation.models import GeneratedAnswer
from app.retrieval.models import RetrievalResult


class GenerationService:
    """
    Generates and validates grounded answers.
    """

    def __init__(
        self,
        llm_service: LLMService,
        context_builder: ContextBuilder,
        citation_validator: CitationValidator,
    ):
        """
        Initialize generation dependencies.
        """

        self.llm_service = llm_service
        self.context_builder = context_builder
        self.citation_validator = citation_validator

    def generate(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> GeneratedAnswer:
        """
        Generate a citation-validated answer.

        Args:
            query:
                User question.

            results:
                Reranked retrieval results.

        Returns:
            Validated generated answer.

        Raises:
            CitationValidationError:
                If the LLM produces invalid citations.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not results:
            raise ValueError("Cannot generate an answer without retrieved evidence.")

        # ----------------------------------------------------
        # Build citation-aware evidence.
        # ----------------------------------------------------

        contexts = self.context_builder.build(results)

        context_text = self.context_builder.format_context(contexts)

        # ----------------------------------------------------
        # Ask Gemini to generate the answer.
        # ----------------------------------------------------

        answer = self.llm_service.generate(
            query=query,
            context=context_text,
        )

        # ----------------------------------------------------
        # Validate citations.
        # ----------------------------------------------------

        self.citation_validator.validate(
            answer=answer,
            contexts=contexts,
        )

        citations = self.citation_validator.extract_citations(answer)

        return GeneratedAnswer(
            answer=answer,
            citations=citations,
        )
