"""
Unit tests for groundedness evaluation.

The real NLI model is not loaded here.
"""

from app.evaluation.groundedness import (
    GroundednessEvaluator,
)
from app.generation.context_builder import (
    CitationContext,
)
from app.retrieval.models import (
    RetrievalResult,
)


class FakeNLIPipeline:
    """
    Fake NLI pipeline returning deterministic entailment.
    """

    def __call__(
        self,
        inputs,
        **kwargs,
    ):
        return [
            {
                "label": "entailment",
                "score": 0.99,
            }
        ]


def create_context(
    citation_id: str = "C1",
    text: str = "Python is a programming language.",
) -> CitationContext:
    """
    Create a citation-aware retrieval context for testing.
    """

    result = RetrievalResult(
        chunk_id="chunk-001",
        document_id="doc-001",
        text=text,
        source="test.pdf",
        page_number=1,
        chunk_index=0,
        score=0.9,
    )

    return CitationContext(
        citation_id=citation_id,
        result=result,
    )


def create_evaluator():
    """
    Create a groundedness evaluator without loading a model.
    """

    evaluator = object.__new__(GroundednessEvaluator)

    evaluator.model_name = "fake-model"
    evaluator.pipeline = FakeNLIPipeline()

    return evaluator


def test_empty_answer_is_not_grounded():
    """
    Empty answers should have groundedness 0.
    """

    evaluator = create_evaluator()

    assert (
        evaluator.evaluate(
            "",
            [create_context()],
        )
        == 0.0
    )


def test_empty_contexts_are_not_grounded():
    """
    Without retrieved contexts, no answer can be grounded.
    """

    evaluator = create_evaluator()

    assert (
        evaluator.evaluate(
            "Python is a language. [C1]",
            [],
        )
        == 0.0
    )


def test_entailing_answer_is_grounded():
    """
    Fake NLI entailment should produce a perfect groundedness
    score.
    """

    evaluator = create_evaluator()

    score = evaluator.evaluate(
        "Python is a language. [C1]",
        [
            create_context(
                citation_id="C1",
                text="Python is a programming language.",
            )
        ],
    )

    assert score == 0.99


def test_extract_citations_preserves_citation_ids():
    """
    Citation extraction must return C1/C2/etc., not only numbers.
    """

    citations = GroundednessEvaluator._extract_citations(
        "Git creates a repository [C1] and sets HEAD [C3]."
    )

    assert citations == ["C1", "C3"]


def test_extract_citations_removes_duplicates():
    """
    Repeated citation IDs in one sentence should only be evaluated once.
    """

    citations = GroundednessEvaluator._extract_citations(
        "This is supported [C1], also [C1]."
    )

    assert citations == ["C1"]


def test_sentence_split_keeps_citation_with_sentence():
    """
    A citation immediately following sentence punctuation must
    remain attached to that sentence.
    """

    sentences = GroundednessEvaluator._split_sentences(
        "Python is a language. [C1] Git is version control. [C2]"
    )

    assert sentences == [
        "Python is a language. [C1]",
        "Git is version control. [C2]",
    ]


def test_directly_supported_git_command_is_grounded(
    monkeypatch,
):
    """A claim directly supported by one evidence sentence should pass."""

    evaluator = GroundednessEvaluator.__new__(GroundednessEvaluator)

    class FakePipeline:
        def __call__(self, payload, **kwargs):
            evidence = payload["text"]

            if "git branch testing" in evidence:
                return {
                    "label": "entailment",
                    "score": 0.99,
                }

            return {
                "label": "neutral",
                "score": 0.80,
            }

    evaluator.pipeline = FakePipeline()

    context = CitationContext(
        citation_id="C1",
        result=RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            source="progit.pdf",
            page_number=71,
            chunk_index=0,
            text=(
                "Creating a New Branch. "
                "You do this with the git branch command: "
                "$ git branch testing. "
                "This creates a new pointer to the same commit."
            ),
            score=1.0,
        ),
    )

    result = evaluator.evaluate(
        answer=(
            "Create a branch without switching using git branch <branchname>. [C1]"
        ),
        contexts=[context],
    )

    assert result.score > 0.5
    assert result.grounded is True
