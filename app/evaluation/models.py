"""
DocsQuery - Evaluation Models

Defines the structured representation of the retrieval
evaluation dataset.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

EvaluationDifficulty = Literal[
    "easy",
    "medium",
    "hard",
    "negative",
    "adversarial",
]


class EvaluationExample(BaseModel):
    """
    Represents one retrieval evaluation question.

    Each example contains:

    - a unique identifier
    - the user query
    - the type/category of the query
    - the difficulty of the query
    - zero or more ground-truth relevant chunks
    - an optional reference answer for answer-level evaluation

    Negative examples intentionally have zero relevant chunks.
    """

    example_id: str

    query: str = Field(
        min_length=1,
    )

    query_type: str = Field(
        min_length=1,
    )

    difficulty: EvaluationDifficulty

    # Positive examples have one or more relevant chunks.
    # Negative examples intentionally have an empty list.
    relevant_chunk_ids: list[str] = Field(
        default_factory=list,
    )

    reference_answer: str | None = None

    @model_validator(mode="after")
    def validate_relevance_annotations(self):
        """
        Ensure relevance annotations agree with difficulty.

        Positive/adversarial examples require at least one
        relevant chunk.

        Negative examples must contain no relevant chunks.
        """

        if self.difficulty == "negative":
            if self.relevant_chunk_ids:
                raise ValueError("Negative examples must have no relevant chunk IDs.")

        elif not self.relevant_chunk_ids:
            raise ValueError(
                "Non-negative examples must have at least one relevant chunk ID."
            )

        return self


class EvaluationDataset(BaseModel):
    """
    Represents the complete retrieval evaluation dataset.
    """

    version: str

    examples: list[EvaluationExample] = Field(
        min_length=1,
    )
