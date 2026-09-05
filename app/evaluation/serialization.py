"""
Utilities for converting evaluation objects into JSON-compatible data.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def to_serializable(value: Any) -> Any:
    """
    Convert common Python model types into JSON-compatible structures.

    Supported forms include:

    - Pydantic models
    - dataclasses
    - dictionaries
    - lists/tuples
    - primitive values
    """

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if is_dataclass(value):
        return asdict(value)

    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]

    return value
