"""
DocsQuery - Evaluation Models

Defines the structured representation of our evaluation
dataset.

An evaluation example contains:

    Query
      +
    Ground-truth relevant chunks
"""

from pydantic import BaseModel, Field


class EvaluationExample(BaseModel):
    """
    One evaluation question and its expected evidence.
    """

    # Unique identifier for this evaluation example.
    example_id: str

    # The question submitted to DocsQuery.
    query: str = Field(
        min_length=1,
    )

    # Chunk IDs that are considered relevant answers/evidence.
    #
    # Multiple chunks are allowed because a question may require
    # information from more than one part of a document.
    relevant_chunk_ids: list[str] = Field(
        min_length=1,
    )


class EvaluationDataset(BaseModel):
    """
    Complete evaluation dataset.
    """

    # Version the dataset so changes are traceable.
    version: str

    # All evaluation questions.
    examples: list[EvaluationExample]
