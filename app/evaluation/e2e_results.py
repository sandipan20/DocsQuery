"""
DocsQuery - End-to-End Evaluation Results

Defines structured results for complete RAG evaluation.

This combines:

    retrieval
    answer generation
    citation quality
    groundedness
    answer correctness
"""

from pydantic import BaseModel


class EndToEndExampleResult(BaseModel):
    """
    Evaluation result for one RAG question.
    """

    example_id: str

    query: str

    # Whether retrieval returned any evidence.
    retrieved: bool

    # Citation metrics.
    citation_valid: bool
    citation_coverage: float

    # Groundedness metrics.
    groundedness_score: float
    grounded: bool

    # Semantic similarity to the reference answer.
    correctness_score: float | None = None

    # The final generated answer is kept for debugging.
    answer: str | None = None


class EndToEndSummary(BaseModel):
    """
    Aggregate results across the complete evaluation dataset.
    """

    dataset_version: str

    num_examples: int

    retrieval_success_rate: float

    citation_validity_rate: float

    average_citation_coverage: float

    average_groundedness: float

    grounded_answer_rate: float

    average_correctness: float | None = None
