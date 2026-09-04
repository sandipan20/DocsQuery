"""
Tests for the complete RAG evaluation pipeline.
"""

from unittest.mock import MagicMock

from app.evaluation.answer_models import (
    AnswerEvaluationResult,
)
from app.evaluation.e2e_evaluator import (
    EndToEndEvaluator,
)
from app.evaluation.models import (
    EvaluationExample,
)
from app.generation.models import GeneratedAnswer
from app.retrieval.models import RetrievalResult


def create_result() -> RetrievalResult:
    """
    Create predictable retrieval evidence.
    """

    return RetrievalResult(
        chunk_id="chunk-001",
        document_id="doc-001",
        text="Git is a distributed version control system.",
        source="progit.pdf",
        page_number=1,
        chunk_index=0,
        score=0.9,
    )


def create_example() -> EvaluationExample:
    """
    Create a predictable evaluation example.
    """

    return EvaluationExample(
        example_id="q001",
        query="What is Git?",
        query_type="repository-basics",
        relevant_chunk_ids=["chunk-001"],
        reference_answer=("Git is a distributed version control system."),
    )


def create_evaluator():
    """
    Create an evaluator with fully mocked dependencies.
    """

    rag_service = MagicMock()

    rag_service.query.return_value = MagicMock(
        generated_answer=GeneratedAnswer(
            answer=("Git is a distributed version control system. [C1]"),
            citations=["C1"],
        ),
        results=[create_result()],
    )

    answer_evaluator = MagicMock()

    answer_evaluator.evaluate.return_value = AnswerEvaluationResult(
        example_id="q001",
        citation_valid=True,
        citation_coverage=1.0,
        grounded=True,
        groundedness_score=0.95,
    )

    correctness_evaluator = MagicMock()

    correctness_evaluator.evaluate.return_value = 0.92

    evaluator = EndToEndEvaluator(
        rag_service=rag_service,
        answer_evaluator=answer_evaluator,
        correctness_evaluator=(correctness_evaluator),
    )

    return evaluator


def test_evaluate_example():
    """
    One complete RAG example should produce structured
    evaluation results.
    """

    evaluator = create_evaluator()

    result = evaluator.evaluate_example(create_example())

    assert result.example_id == "q001"
    assert result.retrieved is True
    assert result.citation_valid is True
    assert result.citation_coverage == 1.0
    assert result.grounded is True
    assert result.groundedness_score == 0.95
    assert result.correctness_score == 0.92
    assert result.answer is not None


def test_evaluate_dataset_produces_summary():
    """
    Dataset evaluation should produce aggregate metrics.
    """

    evaluator = create_evaluator()

    summary, results = evaluator.evaluate(
        examples=[create_example()],
        dataset_version="1.1.0",
    )

    assert summary.dataset_version == "1.1.0"
    assert summary.num_examples == 1
    assert summary.retrieval_success_rate == 1.0
    assert summary.citation_validity_rate == 1.0
    assert summary.average_citation_coverage == 1.0
    assert summary.average_groundedness == 0.95
    assert summary.grounded_answer_rate == 1.0
    assert summary.average_correctness == 0.92

    assert len(results) == 1


def test_empty_dataset_is_rejected():
    """
    Empty evaluation datasets should be rejected.
    """

    evaluator = create_evaluator()

    try:
        evaluator.evaluate(
            examples=[],
            dataset_version="1.1.0",
        )

    except ValueError as exc:
        assert "empty" in str(exc)

    else:
        raise AssertionError("Expected ValueError")
