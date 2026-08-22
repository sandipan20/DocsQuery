"""
Unit tests for the citation-aware context builder.
"""

from app.generation.context_builder import (
    ContextBuilder,
)
from app.retrieval.models import RetrievalResult


def create_result(
    chunk_id: str,
    text: str,
) -> RetrievalResult:
    """
    Create a predictable retrieval result for testing.
    """

    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-001",
        text=text,
        source="test.pdf",
        page_number=1,
        chunk_index=0,
        score=0.9,
    )


def test_context_builder_assigns_citation_ids():
    """
    Each retrieval result should receive a sequential
    citation ID.
    """

    builder = ContextBuilder()

    results = [
        create_result(
            "chunk-1",
            "First evidence.",
        ),
        create_result(
            "chunk-2",
            "Second evidence.",
        ),
    ]

    contexts = builder.build(results)

    assert contexts[0].citation_id == "C1"
    assert contexts[1].citation_id == "C2"


def test_context_builder_preserves_results():
    """
    Citation assignment should not modify the original
    retrieval results.
    """

    builder = ContextBuilder()

    result = create_result(
        "chunk-1",
        "Evidence.",
    )

    contexts = builder.build([result])

    assert contexts[0].result == result


def test_context_is_formatted_with_citations():
    """
    Formatted context should contain the citation ID,
    source, page number, and document text.
    """

    builder = ContextBuilder()

    contexts = builder.build(
        [
            create_result(
                "chunk-1",
                "Python is a programming language.",
            )
        ]
    )

    formatted = builder.format_context(contexts)

    assert "[C1]" in formatted
    assert "test.pdf" in formatted
    assert "Page: 1" in formatted
    assert "Python is a programming language." in formatted


def test_empty_context_returns_empty_string():
    """
    An empty list of evidence should produce an empty
    context string.
    """

    builder = ContextBuilder()

    assert builder.format_context([]) == ""
