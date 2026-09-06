"""
DocsQuery - Evaluation Serialization Helpers

Converts Pydantic models, dataclasses, dictionaries, and sequences
into JSON-compatible Python structures.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def to_serializable(value: Any) -> Any:
    """Convert a supported Python object into JSON-compatible data."""

    # Pydantic v2 models.
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    # Standard-library dataclasses.
    if is_dataclass(value):
        return asdict(value)

    # Recursively convert dictionaries.
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}

    # Recursively convert sequences.
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]

    return value
