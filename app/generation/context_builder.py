"""
DocsQuery - Citation-Aware Context Builder

Converts retrieved chunks into structured context for the LLM.

Each retrieved result receives a stable citation ID:

    [C1]
    [C2]
    [C3]

The LLM can then reference these IDs in its answer.
"""

from dataclasses import dataclass

from app.retrieval.models import RetrievalResult


@dataclass(frozen=True)
class CitationContext:
    """
    Represents one piece of evidence available to the LLM.

    Example:

        citation_id = "C1"
        result = RetrievalResult(...)
    """

    citation_id: str
    result: RetrievalResult


class ContextBuilder:
    """
    Builds citation-aware context from retrieval results.
    """

    def build(
        self,
        results: list[RetrievalResult],
    ) -> list[CitationContext]:
        """
        Assign sequential citation IDs to retrieval results.

        Example:

            result 1 → C1
            result 2 → C2
            result 3 → C3

        Args:
            results:
                Reranked retrieval results.

        Returns:
            Citation-aware evidence.
        """

        return [
            CitationContext(
                citation_id=f"C{index}",
                result=result,
            )
            for index, result in enumerate(
                results,
                start=1,
            )
        ]

    def format_context(
        self,
        contexts: list[CitationContext],
    ) -> str:
        """
        Convert citation contexts into text for the LLM.

        Each evidence block contains:

            Citation ID
            Source
            Page
            Content
        """

        sections = []

        for context in contexts:
            result = context.result

            sections.append(
                f"[{context.citation_id}]\n"
                f"Source: {result.source}\n"
                f"Page: {result.page_number}\n"
                f"Content:\n"
                f"{result.text}"
            )

        return "\n\n".join(sections)
