"""
DocsQuery - Answer Evaluation Models

Structured models used to evaluate generated RAG answers.

The retrieval evaluation asks:

    "Did we retrieve the right evidence?"

The answer evaluation asks:

    "Did the generated answer use that evidence correctly?"
"""

from pydantic import BaseModel, Field


class AnswerEvaluationExample(BaseModel):
    """
    Ground-truth information for evaluating one answer.
    """

    # Evaluation example ID.
    example_id: str

    # User's original question.
    query: str

    # Ground-truth relevant chunks.
    relevant_chunk_ids: list[str] = Field(
        min_length=1,
    )

    # Optional reference answer.
    #
    # We will use this for answer-correctness evaluation.
    reference_answer: str | None = None


class AnswerEvaluationResult(BaseModel):
    """
    Evaluation result for one generated answer.
    """

    example_id: str

    # Whether all cited IDs actually exist in the evidence.
    citation_valid: bool

    # Fraction of answer sentences containing citations.
    citation_coverage: float

    # Whether the answer could be generated from the
    # provided evidence.
    grounded: bool
    groundedness_score: float = 0.0
    # Optional correctness score.
    #
    # This will later be produced by a separate evaluator.
    correctness_score: float | None = None
