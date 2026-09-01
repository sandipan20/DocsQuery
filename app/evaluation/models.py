"""
DocsQuery - Evaluation Models

Defines the structured representation of the retrieval
evaluation dataset.
"""

from pydantic import BaseModel, Field


class EvaluationExample(BaseModel):
    """
    Represents one retrieval evaluation question.

    Each example contains:

    - a unique identifier
    - the user query
    - the type/category of the query
    - one or more ground-truth relevant chunks
    """

    # Unique identifier for this evaluation example.
    example_id: str

    # The natural-language question sent to the retriever.
    query: str = Field(
        min_length=1,
    )

    # Category of the query.
    #
    # This helps us later determine whether retrieval performs
    # differently for branching, merging, rebasing, etc.
    query_type: str = Field(
        min_length=1,
    )

    # Chunk IDs that are considered relevant for this query.
    #
    # More than one chunk is allowed because some questions
    # require evidence from multiple sections of the corpus.
    relevant_chunk_ids: list[str] = Field(
        min_length=1,
    )


class EvaluationDataset(BaseModel):
    """
    Represents the complete retrieval evaluation dataset.
    """

    # Dataset version. This changes when the dataset schema
    # or ground-truth examples are meaningfully updated.
    version: str

    # All evaluation questions.
    examples: list[EvaluationExample] = Field(
        min_length=1,
    )
