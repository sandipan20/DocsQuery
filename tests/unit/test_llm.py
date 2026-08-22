"""
Unit tests for the Gemini LLM service.
"""

from unittest.mock import MagicMock

import pytest

from app.generation.llm import LLMService


def create_service():
    """
    Create an LLM service with a fake Gemini client.

    No real API request is made.
    """

    service = object.__new__(LLMService)

    service.model = "test-model"
    service.temperature = 0.0
    service.max_tokens = 1000

    service.client = MagicMock()

    return service


def test_empty_query_is_rejected():
    """
    Empty queries should not be sent to Gemini.
    """

    service = create_service()

    with pytest.raises(ValueError):
        service.generate(
            "",
            "Evidence",
        )


def test_empty_context_is_rejected():
    """
    Gemini should never be called without evidence.
    """

    service = create_service()

    with pytest.raises(ValueError):
        service.generate(
            "What is Python?",
            "",
        )


def test_generate_returns_gemini_response():
    """
    Verify that Gemini response text is returned.
    """

    service = create_service()

    response = MagicMock()

    response.text = "Python is a programming language. [C1]"

    service.client.models.generate_content.return_value = response

    answer = service.generate(
        query="What is Python?",
        context=("[C1]\nPython is a programming language."),
    )

    assert answer == ("Python is a programming language. [C1]")

    service.client.models.generate_content.assert_called_once()


def test_empty_gemini_response_raises_error():
    """
    Empty Gemini responses should fail clearly.
    """

    service = create_service()

    response = MagicMock()

    response.text = ""

    service.client.models.generate_content.return_value = response

    with pytest.raises(RuntimeError):
        service.generate(
            query="What is Python?",
            context="[C1] Python is a language.",
        )
