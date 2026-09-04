"""
DocsQuery - Evaluation Context Helpers

Provides a small helper for converting retrieval results into
the citation-aware context objects used by answer evaluation.

The ordering must remain identical to the production
generation pipeline:

    results[0] -> C1
    results[1] -> C2
    ...
"""

from app.generation.context_builder import (
    CitationContext,
    ContextBuilder,
)
from app.retrieval.models import RetrievalResult


def build_evaluation_contexts(
    results: list[RetrievalResult],
) -> list[CitationContext]:
    """
    Convert retrieval results into citation contexts.

    Args:
        results:
            Reranked retrieval results.

    Returns:
        Citation-aware contexts.
    """

    builder = ContextBuilder()

    return builder.build(results)
