"""
DocsQuery - Generation Models

Models representing generated answers and citation metadata.
"""

from pydantic import BaseModel, Field


class GeneratedAnswer(BaseModel):
    """
    Represents an answer generated from retrieved evidence.
    """

    answer: str

    citations: list[str] = Field(default_factory=list)
